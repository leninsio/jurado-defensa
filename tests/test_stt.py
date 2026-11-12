"""Tests for app.stt: ffmpeg normalization plus Spanish whisper transcription."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import stt  # noqa: E402
from app.stt import SttError, transcribe_audio, transcribe_segments  # noqa: E402

SPANISH_VOICE = "Paulina (Español (México))"


def _say_clip(path: Path, text: str = "hola jurado, este es mi proyecto") -> Path:
    subprocess.run(
        ["say", "-v", SPANISH_VOICE, "-o", str(path), "--", text],
        check=True,
    )
    return path


def _two_sentence_clip(path: Path) -> Path:
    return _say_clip(
        path,
        "Hola jurado. [[slnc 900]] Este es mi proyecto de grado.",
    )


def _tone(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4",
            "-ar", "16000", "-ac", "1", str(path),
        ],
        check=True,
    )
    return path


def _silence(path: Path, seconds: float = 1.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", str(seconds), str(path),
        ],
        check=True,
    )
    return path


def test_transcribes_spanish_say_clip(tmp_path: Path) -> None:
    clip = _say_clip(tmp_path / "clip.aiff")

    text = transcribe_audio(clip).lower()

    assert "jurado" in text
    assert "proyecto" in text


def test_transcribes_browser_webm_opus_clip(tmp_path: Path) -> None:
    clip = _say_clip(tmp_path / "clip.aiff")
    webm = tmp_path / "clip.webm"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
            "-c:a", "libopus", "-b:a", "32k", str(webm),
        ],
        check=True,
    )

    text = transcribe_audio(webm).lower()

    assert "jurado" in text
    assert "proyecto" in text


def test_silent_audio_returns_empty_string(tmp_path: Path) -> None:
    assert transcribe_audio(_silence(tmp_path / "silence.wav")) == ""


def test_missing_model_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JURADO_WHISPER_MODEL", str(tmp_path / "missing-model.bin"))

    with pytest.raises(SttError, match="model"):
        transcribe_audio(_silence(tmp_path / "silence.wav"))


def test_missing_ffmpeg_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = _silence(tmp_path / "silence.wav")
    monkeypatch.setenv("PATH", "")

    with pytest.raises(SttError, match="ffmpeg"):
        transcribe_audio(clip)


def test_missing_audio_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SttError):
        transcribe_audio(tmp_path / "nope.wav")


def test_transcribe_segments_returns_increasing_timestamps(tmp_path: Path) -> None:
    clip = _two_sentence_clip(tmp_path / "two-sentences.aiff")

    segments = transcribe_segments(clip)

    assert segments, "two spoken sentences should produce at least one segment"
    starts = [start for start, _, _ in segments]
    assert starts == sorted(starts)
    assert all(end > start for start, end, _ in segments)
    assert segments[0][0] >= 0
    assert segments[-1][1] > segments[0][0]
    text = " ".join(item for _, _, item in segments).lower()
    assert "jurado" in text
    assert "proyecto" in text


def test_transcribe_segments_returns_empty_for_silence(tmp_path: Path) -> None:
    assert transcribe_segments(_silence(tmp_path / "silence.wav")) == []


class _SegmentModel:
    """Whisper stand-in that returns fixed segments in centiseconds."""

    def __init__(self, segments: list[SimpleNamespace]) -> None:
        self.segments = segments

    def transcribe(self, media: str) -> list[SimpleNamespace]:
        return self.segments


def test_transcribe_segments_converts_centiseconds_to_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"not-a-real-model")
    monkeypatch.setenv("JURADO_WHISPER_MODEL", str(model_file))

    fake = _SegmentModel(
        [
            SimpleNamespace(t0=150, t1=275, text=" hola "),
            SimpleNamespace(t0=275, t1=400, text=" mundo "),
            SimpleNamespace(t0=400, t1=450, text="   "),
        ]
    )
    monkeypatch.setattr(stt, "_load_model", lambda *args, **kwargs: fake)
    monkeypatch.setattr(stt, "_model", None)
    monkeypatch.setattr(stt, "_model_loaded_from", None)

    clip = _tone(tmp_path / "tone.wav")

    assert transcribe_segments(clip) == [(1.5, 2.75, "hola"), (2.75, 4.0, "mundo")]


class _FakeModel:
    """Records overlapping inference calls to prove serialization."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    def transcribe(self, media: str) -> list[SimpleNamespace]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(0.2)
        self.active -= 1
        return [SimpleNamespace(text=" hola ")]


def test_concurrent_calls_are_serialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"not-a-real-model")
    monkeypatch.setenv("JURADO_WHISPER_MODEL", str(model_file))

    fake = _FakeModel()
    monkeypatch.setattr(stt, "_load_model", lambda *args, **kwargs: fake)
    monkeypatch.setattr(stt, "_model", None)
    monkeypatch.setattr(stt, "_model_loaded_from", None)

    clip = _tone(tmp_path / "tone.wav")
    results: list[str] = []

    def work() -> None:
        results.append(transcribe_audio(clip))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["hola", "hola"]
    assert fake.max_active == 1
