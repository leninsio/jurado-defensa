"""Speech-to-text for the local jury simulator.

Browser recordings (webm/opus) and any other ffmpeg-readable audio are
normalized to 16 kHz mono wav and transcribed in Spanish with whisper.cpp
through pywhispercpp. The model is loaded lazily, once, and every
transcription is serialized by a lock because a whisper.cpp context is not
safe for concurrent use.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from array import array
from pathlib import Path

DEFAULT_MODEL_PATH = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Jarvis"
    / "models"
    / "ggml-large-v3-turbo-q5_0.bin"
)
MODEL_ENV_VAR = "JURADO_WHISPER_MODEL"
LANGUAGE = "es"
SAMPLE_RATE = 16000
SILENCE_PEAK_THRESHOLD = 300  # 16-bit samples; below this the clip is treated as silence


class SttError(RuntimeError):
    """Raised when a clip cannot be transcribed."""


_model = None
_model_loaded_from: Path | None = None
_model_lock = threading.Lock()


def model_path() -> Path:
    """Return the whisper model path, honoring the environment override."""
    override = os.environ.get(MODEL_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_MODEL_PATH


def transcribe_audio(path: str | Path) -> str:
    """Transcribe an audio file to Spanish text.

    Accepts any audio format ffmpeg can decode. Returns an empty string for
    empty or silent audio.
    """
    segments = _transcribe_raw(path)
    return " ".join((segment.text or "").strip() for segment in segments).strip()


def transcribe_segments(path: str | Path) -> list[tuple[float, float, str]]:
    """Transcribe an audio file into ``(start_s, end_s, text)`` segments.

    Uses the same ffmpeg normalization, model and lock as
    :func:`transcribe_audio`. pywhispercpp exposes whisper.cpp timestamps in
    centiseconds (1/100 s), so they are divided by 100 here. Silent or empty
    audio yields an empty list.
    """
    result: list[tuple[float, float, str]] = []
    for segment in _transcribe_raw(path):
        text = (segment.text or "").strip()
        if not text:
            continue
        start = max(0.0, float(getattr(segment, "t0", 0) or 0) / 100.0)
        end = max(start, float(getattr(segment, "t1", 0) or 0) / 100.0)
        result.append((start, end, text))
    return result


def _transcribe_raw(path: str | Path) -> list:
    """Run whisper once and return its raw segment objects."""
    audio = Path(path)
    if not audio.is_file():
        raise SttError(f"Audio file not found: {audio}")

    _require_ffmpeg()
    model_file = model_path()
    if not model_file.is_file():
        raise SttError(
            f"Whisper model not found: {model_file}. "
            f"Set {MODEL_ENV_VAR} to a ggml model path."
        )

    with _model_lock:
        with tempfile.TemporaryDirectory(prefix="jurado-stt-") as tmp_dir:
            wav_path = Path(tmp_dir) / "audio.wav"
            _convert_to_wav(audio, wav_path)
            if _is_silent(wav_path):
                return []
            model = _get_model(model_file)
            return list(model.transcribe(str(wav_path)))


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SttError(
            "ffmpeg not found on PATH; install it with `brew install ffmpeg`."
        )


def _convert_to_wav(source: Path, target: Path) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "wav",
        str(target),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise SttError(f"ffmpeg could not be executed: {exc}") from exc

    if result.returncode != 0 or not target.is_file():
        detail = (result.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit code {result.returncode}"
        raise SttError(f"ffmpeg could not decode {source}: {reason}")


def _is_silent(wav_path: Path) -> bool:
    try:
        with wave.open(str(wav_path)) as wav:
            frame_count = wav.getnframes()
            if frame_count == 0:
                return True
            frames = wav.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise SttError(f"ffmpeg produced an unreadable wav file: {exc}") from exc

    samples = array("h")
    samples.frombytes(frames)
    if not samples:
        return True
    return max(abs(sample) for sample in samples) < SILENCE_PEAK_THRESHOLD


def _get_model(model_file: Path):
    """Load the whisper model once per resolved model path."""
    global _model, _model_loaded_from

    if _model is None or _model_loaded_from != model_file:
        _model = _load_model(model_file)
        _model_loaded_from = model_file
    return _model


def _load_model(model_file: Path):
    from pywhispercpp.model import Model

    return Model(
        str(model_file),
        language=LANGUAGE,
        print_progress=False,
        suppress_non_speech_tokens=True,
        redirect_whispercpp_logs_to=None,
    )
