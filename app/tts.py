"""Text-to-speech for the local jury simulator.

Each jury role gets a distinct configured voice. Synthesis runs through one
serial priority queue: a live jury turn always jumps ahead of background
prefetch or filler work, even when the engine is a single long-lived XTTS
worker (one model load, one stdin/stdout protocol stream).
"""

from __future__ import annotations

import heapq
import re
import os
import shutil
import subprocess
import time
import uuid
import wave
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Callable

VOICES: dict[str, str] = {
    "juez_metodologo": "Paulina (Español (México))",
    "juez_tecnico": "Eddy (Español (México))",
    "juez_presidente": "Mónica",
    "tutor": "Grandpa (Español (México))",
    "default": "Sandy (Español (México))",
}

# Words per minute per role; `say` accepts 80-500.
RATES: dict[str, int] = {
    "juez_metodologo": 175,
    "juez_tecnico": 185,
    "juez_presidente": 160,
    "tutor": 170,
    "default": 175,
}

FILE_FORMAT = "WAVE"
DATA_FORMAT = "LEI16@22050"

# Synthesis priorities: lower runs first. Live turns beat every background job,
# prefetched judge questions and warm-up clips wait for idle time.
PRIORITY_HIGH = 0
PRIORITY_NORMAL = 10
PRIORITY_LOW = 20


class TtsError(RuntimeError):
    """Raised when speech cannot be synthesized."""


FALLBACK_VOICES: dict[str, str] = {}


def configure_voices(roles: dict[str, dict]) -> None:
    """Override VOICES with each role's ``voice`` from config.json (single source)."""
    for role, settings in roles.items():
        voice = settings.get("voice")
        if voice:
            VOICES[role] = voice
        fallback = settings.get("voice_fallback")
        if fallback:
            FALLBACK_VOICES[role] = fallback


def voice_for(role: str) -> str:
    """Return the voice assigned to a role, falling back to the default."""
    return VOICES.get(role, VOICES["default"])


def speakable_text(text: str) -> str:
    """Strip Markdown markers (headings, bullets, emphasis, code) so `say` does not read them."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(#{1,6}\s+|[-*+]\s+|\d+\.\s+(?=\D))", "", line)
        line = re.sub(r"(\*\*|__|\*|_|`)(.+?)\1", r"\2", line)
        lines.append(line.replace("`", "").strip())
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Sentence streaming (S19): one judge reply becomes short spoken chunks
# ---------------------------------------------------------------------------

# A chunk shorter than this is a fragment ("Sí.", "Claro.") and is merged into
# the next sentence so the voice does not say it in isolation.
MIN_CHUNK_CHARS = 12
# Longer chunks delay playback; two short sentences are one natural breath.
MAX_CHUNK_CHARS = 220
MAX_SENTENCES_PER_CHUNK = 2
# The very first chunk is what the student hears first, so a long opening
# sentence is cut at a clause (or space) boundary instead of waiting for all of
# it: ~80 characters is about four seconds of speech.
MAX_FIRST_CHUNK_CHARS = 80

_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_BREAK = re.compile(r"[,;:—]\s")


def _cap_first_sentence(sentence: str) -> tuple[str, str]:
    """Split an opening sentence so the first spoken chunk stays short."""
    if len(sentence) <= MAX_FIRST_CHUNK_CHARS:
        return sentence, ""
    window_end = int(MAX_FIRST_CHUNK_CHARS * 1.25)
    window = sentence[:window_end]
    clauses = [match.end() for match in _CLAUSE_BREAK.finditer(window)]
    if clauses:
        cut = clauses[-1]
    else:
        space = sentence[:MAX_FIRST_CHUNK_CHARS].rfind(" ")
        cut = space if space > 0 else MAX_FIRST_CHUNK_CHARS
    head = sentence[:cut].strip()
    tail = sentence[cut:].strip()
    if not head or not tail:
        return sentence, ""
    return head, tail


def split_speech_chunks(text: str) -> list[str]:
    """Split a reply into playable chunks, first chunk as short as possible.

    The first sentence stands alone so playback starts with the judge's short
    reaction ("Bien, me queda claro.") while the rest is still being
    synthesized. Later sentences group one or two per chunk. No words are lost:
    joining the chunks with spaces reproduces the speakable text.
    """
    clean = " ".join(speakable_text(text or "").split())
    if not clean:
        return []

    sentences = [piece.strip() for piece in _SENTENCE_BREAK.split(clean) if piece.strip()]

    merged: list[str] = []
    pending = ""
    for sentence in sentences:
        piece = f"{pending} {sentence}".strip() if pending else sentence
        if len(piece) < MIN_CHUNK_CHARS:
            pending = piece
            continue
        merged.append(piece)
        pending = ""
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]} {pending}".strip()
        else:
            merged.append(pending)

    if merged:
        head, tail = _cap_first_sentence(merged[0])
        merged = [head] + ([tail] if tail else []) + merged[1:]

    chunks: list[str] = []
    current: list[str] = []
    for index, sentence in enumerate(merged):
        current.append(sentence)
        length = len(" ".join(current))
        if index == 0 or len(current) >= MAX_SENTENCES_PER_CHUNK or length > MAX_CHUNK_CHARS:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Serial priority queue
# ---------------------------------------------------------------------------


class _Job:
    __slots__ = ("key", "work", "priority", "sequence", "future", "started")

    def __init__(
        self,
        key: str,
        work: Callable[[], object],
        priority: int,
        sequence: int,
    ) -> None:
        self.key = key
        self.work = work
        self.priority = priority
        self.sequence = sequence
        self.future: Future = Future()
        self.started = False


class PrioritySerialRunner:
    """Run submitted callables one at a time, highest priority first.

    Jobs are unique per key: submitting the same key again returns the existing
    future and, while the job is still queued, raises its priority. That is how
    a consumed prefetched question jumps the queue without a second synthesis.
    """

    def __init__(self, name: str = "tts") -> None:
        self.name = name
        self._condition = threading.Condition()
        self._heap: list[tuple[int, int, _Job]] = []
        self._jobs: dict[str, _Job] = {}
        self._sequence = 0
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(
        self, key: str, work: Callable[[], object], priority: int = PRIORITY_NORMAL
    ) -> Future:
        with self._condition:
            existing = self._jobs.get(key)
            if existing is not None:
                if not existing.started and priority < existing.priority:
                    existing.priority = priority
                    self._rebuild_locked()
                    self._condition.notify()
                return existing.future
            job = _Job(key, work, priority, self._sequence)
            self._sequence += 1
            self._jobs[key] = job
            heapq.heappush(self._heap, (job.priority, job.sequence, job))
            self._start_locked()
            self._condition.notify()
            return job.future

    def _rebuild_locked(self) -> None:
        self._heap = [(job.priority, job.sequence, job) for _, _, job in self._heap]
        heapq.heapify(self._heap)

    def _start_locked(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name=f"{self.name}-runner", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._heap and not self._closed:
                    self._condition.wait()
                if not self._heap:
                    return
                _, _, job = heapq.heappop(self._heap)
                job.started = True
            try:
                result = job.work()
            except BaseException as exc:  # noqa: BLE001 - the future carries it
                if not job.future.cancelled():
                    job.future.set_exception(exc)
            else:
                if not job.future.cancelled():
                    job.future.set_result(result)
            finally:
                with self._condition:
                    if self._jobs.get(job.key) is job:
                        self._jobs.pop(job.key, None)


_runner = PrioritySerialRunner("tts")


def new_audio_path(role: str, out_dir: str | Path) -> Path:
    """Allocate a unique wav path before synthesis starts (streamed chunks)."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slug(role)}-{uuid.uuid4().hex}.wav"


def request_synthesis(
    text: str,
    role: str,
    out_dir: str | Path,
    target: str | Path,
    priority: int = PRIORITY_NORMAL,
) -> Future:
    """Queue ``text`` into an allocated path and return when the file exists.

    Idempotent per path: an existing non-empty file resolves immediately, and a
    still-queued job for the same path is promoted instead of duplicated.
    """
    if not text or not text.strip():
        raise TtsError("Cannot synthesize empty text")
    target = Path(target)
    if target.is_file() and target.stat().st_size > 0:
        ready: Future = Future()
        ready.set_result(target)
        return ready

    def work() -> Path:
        return _render_role_now(text, role, Path(out_dir), target)

    return _runner.submit(str(target), work, priority)


def synthesize_to(
    text: str,
    role: str,
    out_dir: str | Path,
    target: str | Path,
    priority: int = PRIORITY_NORMAL,
) -> Path:
    """Blocking variant of :func:`request_synthesis`."""
    return request_synthesis(text, role, out_dir, target, priority=priority).result()


PIPER_PREFIX = "piper:"
PIPER_VOICES_DIR = Path(
    os.environ.get(
        "JURADO_PIPER_VOICES",
        Path.home() / "Library" / "Application Support" / "JuradoDefensa" / "voices",
    )
)
_piper_cache: dict[str, object] = {}
_piper_lock = threading.Lock()


def parse_piper_voice(voice: str) -> tuple[str, int | None] | None:
    """Parse ``piper:<model>[#speaker]``; return None for a macOS `say` voice."""
    if not voice.startswith(PIPER_PREFIX):
        return None
    model, _, speaker = voice[len(PIPER_PREFIX):].partition("#")
    return model, int(speaker) if speaker else None


def _synthesize_piper(text: str, model: str, speaker: int | None, output_path: Path) -> Path:
    model_path = PIPER_VOICES_DIR / f"{model}.onnx"
    if not model_path.is_file():
        raise TtsError(f"Piper voice not found: {model_path}")
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise TtsError("piper-tts is not installed") from exc
    with _piper_lock:
        voice = _piper_cache.get(model)
        if voice is None:
            voice = PiperVoice.load(str(model_path))
            _piper_cache[model] = voice
        try:
            with wave.open(str(output_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=SynthesisConfig(speaker_id=speaker))
        except Exception as exc:  # noqa: BLE001 - surface any engine failure as TtsError
            raise TtsError(f"Piper failed for {model}: {exc}") from exc
    return output_path


XTTS_PREFIX = "xtts:"
XTTS_HOME = Path(
    os.environ.get(
        "JURADO_XTTS_HOME",
        Path.home() / "Library" / "Application Support" / "JuradoDefensa" / "engines" / "xtts",
    )
)
XTTS_WORKER = Path(__file__).resolve().parents[1] / "scripts" / "xtts_worker.py"
XTTS_START_TIMEOUT = float(os.environ.get("JURADO_XTTS_START_TIMEOUT", "180"))
_xtts_lock = threading.Lock()
_xtts_process: subprocess.Popen | None = None


def parse_xtts_voice(voice: str) -> str | None:
    """Return the XTTS speaker of ``xtts:<speaker>``; None for other engines."""
    if not voice.startswith(XTTS_PREFIX):
        return None
    return voice[len(XTTS_PREFIX):].strip() or None


def _xtts_worker() -> subprocess.Popen:
    """Start (once) the long-lived XTTS worker in its own engine venv."""
    global _xtts_process
    if _xtts_process is not None and _xtts_process.poll() is None:
        return _xtts_process
    python = XTTS_HOME / "venv" / "bin" / "python"
    if not python.is_file():
        raise TtsError(f"XTTS engine not installed: {python}")
    process = subprocess.Popen(
        [str(python), str(XTTS_WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    ready = _read_json_line(process, XTTS_START_TIMEOUT)
    if not ready.get("ready"):
        process.kill()
        raise TtsError(f"XTTS worker did not start: {ready}")
    with _xtts_lock:
        _xtts_process = process
    return process


def _read_json_line(process: subprocess.Popen, timeout: float) -> dict:
    import json
    import select

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            raise TtsError("XTTS worker timed out")
        ready, _, _ = select.select([process.stdout], [], [], remaining)
        if not ready:
            continue
        line = process.stdout.readline()
        if not line:
            raise TtsError("XTTS worker exited")
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)


def _synthesize_xtts(text: str, speaker: str, output_path: Path) -> Path:
    """One request to the shared XTTS worker; callers run inside the runner."""
    import json

    global _xtts_process
    try:
        process = _xtts_worker()
        process.stdin.write(
            json.dumps({"text": text, "speaker": speaker, "out": str(output_path)}, ensure_ascii=False)
            + "\n"
        )
        process.stdin.flush()
        reply = _read_json_line(process, 120)
    except (OSError, ValueError) as exc:
        with _xtts_lock:
            _xtts_process = None
        raise TtsError(f"XTTS worker failed: {exc}") from exc
    if not reply.get("ok"):
        raise TtsError(f"XTTS failed: {reply.get('error')}")
    return output_path


def warm_up_xtts() -> None:
    """Load the XTTS model ahead of time when any role uses it."""
    if any(parse_xtts_voice(v) for v in VOICES.values()):
        _runner.submit("xtts:worker", _xtts_worker, PRIORITY_NORMAL).result()


def stop_xtts() -> None:
    global _xtts_process
    with _xtts_lock:
        process = _xtts_process
        if process is not None and process.poll() is None:
            process.terminate()
        _xtts_process = None


def synthesize(text: str, role: str, out_dir: str | Path, priority: int = PRIORITY_NORMAL) -> Path:
    """Synthesize ``text`` with the role's voice; on failure use its fallback voice."""
    if not text or not text.strip():
        raise TtsError("Cannot synthesize empty text")
    target = new_audio_path(role, out_dir)
    return _request_role(text, role, Path(out_dir), target, priority).result()


def _request_role(
    text: str,
    role: str,
    out_dir: Path,
    target: Path,
    priority: int,
) -> Future:
    if target.is_file() and target.stat().st_size > 0:
        ready: Future = Future()
        ready.set_result(target)
        return ready

    def work() -> Path:
        return _render_role_now(text, role, out_dir, target)

    return _runner.submit(str(target), work, priority)


def synthesize_with_voice(
    text: str,
    voice: str,
    out_dir: str | Path,
    role: str = "default",
    priority: int = PRIORITY_NORMAL,
) -> Path:
    """Synthesize with an explicit voice: ``piper:<model>[#speaker]`` or a macOS `say` voice."""
    if not text or not text.strip():
        raise TtsError("Cannot synthesize empty text")
    directory = Path(out_dir)
    target = new_audio_path(role, directory)

    def work() -> Path:
        return _render_voice_now(text, voice, directory, role, target)

    return _runner.submit(str(target), work, priority).result()


def _render_role_now(text: str, role: str, out_dir: Path, target: Path) -> Path:
    """Render with the role voice and its configured fallback (inside the runner)."""
    try:
        return _render_voice_now(text, voice_for(role), out_dir, role, target)
    except TtsError:
        fallback = FALLBACK_VOICES.get(role)
        if not fallback or fallback == voice_for(role):
            raise
        return _render_voice_now(text, fallback, out_dir, role, target)


def _render_voice_now(
    text: str, voice: str, out_dir: Path, role: str, target: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    xtts_speaker = parse_xtts_voice(voice)
    if xtts_speaker is not None:
        return _synthesize_xtts(speakable_text(text) or text, xtts_speaker, target)

    piper_voice = parse_piper_voice(voice)
    if piper_voice is not None:
        return _synthesize_piper(speakable_text(text) or text, *piper_voice, target)

    say_binary = shutil.which("say")
    if say_binary is None:
        raise TtsError("macOS `say` not found on PATH")

    command = [
        say_binary,
        "-v",
        voice,
        "--file-format",
        FILE_FORMAT,
        "--data-format",
        DATA_FORMAT,
        "-r",
        str(RATES.get(role, RATES["default"])),
        "-o",
        str(target),
        "--",
        speakable_text(text) or text,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise TtsError(f"`say` could not be executed: {exc}") from exc

    if (
        result.returncode != 0
        or not Path(target).is_file()
        or Path(target).stat().st_size == 0
    ):
        detail = (result.stderr or "").strip() or f"exit code {result.returncode}"
        raise TtsError(f"`say` failed for voice {voice!r}: {detail}")

    return target


def _slug(role: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", role).strip("_")
    return (slug or "voice")[:40]
