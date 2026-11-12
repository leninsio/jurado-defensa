"""Tests for app.tts: macOS `say` synthesis to browser-playable wav."""

from __future__ import annotations

import sys
import threading
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tts import (  # noqa: E402
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    PrioritySerialRunner,
    RATES,
    VOICES,
    TtsError,
    split_speech_chunks,
    synthesize,
    synthesize_to,
)

REQUIRED_ROLES = ["juez_metodologo", "juez_tecnico", "juez_presidente", "tutor"]


def test_voices_cover_every_role_with_distinct_voices() -> None:
    for role in [*REQUIRED_ROLES, "default"]:
        assert role in VOICES
        assert VOICES[role].strip()

    judge_voices = {VOICES[role] for role in REQUIRED_ROLES}
    assert len(judge_voices) == len(REQUIRED_ROLES)


def test_rates_are_within_say_range() -> None:
    for role, rate in RATES.items():
        assert role in VOICES
        assert 80 <= rate <= 500


def test_synthesize_writes_browser_playable_wav(tmp_path: Path) -> None:
    out = synthesize("Hola jurado, comencemos la defensa.", "juez_metodologo", tmp_path)

    assert isinstance(out, Path)
    assert out.exists()
    assert out.parent == tmp_path
    assert out.stat().st_size > 1000

    with wave.open(str(out)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() > 0


def test_synthesize_uses_unique_filenames(tmp_path: Path) -> None:
    first = synthesize("primera intervencion del tutor", "tutor", tmp_path)
    second = synthesize("primera intervencion del tutor", "tutor", tmp_path)

    assert first != second
    assert first.exists() and second.exists()


def test_synthesize_unknown_role_falls_back_to_default(tmp_path: Path) -> None:
    out = synthesize("hola", "rol_desconocido", tmp_path)

    assert out.exists()


def test_synthesize_empty_text_raises(tmp_path: Path) -> None:
    with pytest.raises(TtsError):
        synthesize("   ", "tutor", tmp_path)


def test_synthesize_missing_say_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", "")

    with pytest.raises(TtsError, match="say"):
        synthesize("hola", "tutor", tmp_path)


def test_markdown_is_removed_before_speaking():
    from app.tts import speakable_text

    raw = "## Respuesta\nLa empresa tiene **tres (3) trabajadores**:\n- *Gerente* General\n- `Cajero`"
    assert speakable_text(raw) == "Respuesta\nLa empresa tiene tres (3) trabajadores:\nGerente General\nCajero"


def test_piper_voice_writes_a_playable_wav(tmp_path):
    import wave

    from app import tts

    if not (tts.PIPER_VOICES_DIR / "es_MX-claude-high.onnx").is_file():
        import pytest

        pytest.skip("Piper voice not downloaded")
    path = tts.synthesize_with_voice("Adelante, ya puedes empezar.", "piper:es_MX-claude-high", tmp_path)
    with wave.open(str(path)) as wav:
        assert wav.getnframes() / wav.getframerate() > 0.5


def test_piper_speaker_suffix_is_parsed():
    from app.tts import parse_piper_voice

    assert parse_piper_voice("piper:es_ES-sharvard-medium#1") == ("es_ES-sharvard-medium", 1)
    assert parse_piper_voice("piper:es_MX-ald-medium") == ("es_MX-ald-medium", None)
    assert parse_piper_voice("Paulina (Español (México))") is None


def test_xtts_voice_is_parsed():
    from app.tts import parse_xtts_voice

    assert parse_xtts_voice("xtts:Claribel Dervla") == "Claribel Dervla"
    assert parse_xtts_voice("piper:es_MX-ald-medium") is None


def test_xtts_failure_falls_back_to_the_role_fallback_voice(tmp_path, monkeypatch):
    import wave

    from app import tts

    if not (tts.PIPER_VOICES_DIR / "es_MX-ald-medium.onnx").is_file():
        import pytest

        pytest.skip("Piper voice not downloaded")

    def broken(*_args, **_kwargs):
        raise tts.TtsError("xtts down")

    monkeypatch.setattr(tts, "_synthesize_xtts", broken)
    monkeypatch.setattr(tts, "VOICES", {**tts.VOICES, "juez_tecnico": "xtts:Royston Min"})
    monkeypatch.setattr(tts, "FALLBACK_VOICES", {"juez_tecnico": "piper:es_MX-ald-medium"})
    path = tts.synthesize("Buenas tardes.", "juez_tecnico", tmp_path)
    with wave.open(str(path)) as wav:
        assert wav.getnframes() > 0


# ---------------------------------------------------------------------------
# S19 sentence streaming: split one judge reply into short spoken chunks
# ---------------------------------------------------------------------------


def test_split_speech_chunks_starts_with_the_short_reaction() -> None:
    text = (
        "Bien, me queda claro. ¿Por qué elegiste un sistema web en lugar de "
        "seguir con Excel? ¿Cómo lo validaste con la empresa?"
    )

    chunks = split_speech_chunks(text)

    assert chunks[0] == "Bien, me queda claro."
    assert all(chunk.strip() for chunk in chunks)
    assert " ".join(chunks) == " ".join(text.split())


def test_split_speech_chunks_merges_short_fragments() -> None:
    text = "Sí. ¿Por qué elegiste ese enfoque para el proyecto de grado?"

    chunks = split_speech_chunks(text)

    assert chunks == ["Sí. ¿Por qué elegiste ese enfoque para el proyecto de grado?"]


def test_split_speech_chunks_groups_one_or_two_sentences() -> None:
    text = (
        "Entiendo, gracias. Primero quiero saber cómo detectaste el problema. "
        "Después, cómo recogiste la información. Y al final, qué encontraste."
    )

    chunks = split_speech_chunks(text)

    assert chunks[0] == "Entiendo, gracias."
    assert len(chunks) == 3
    assert all(len(chunk.split(".")) - 1 <= 2 for chunk in chunks)


def test_split_speech_chunks_keeps_a_single_sentence_whole() -> None:
    assert split_speech_chunks("¿Cómo validaste la propuesta?") == [
        "¿Cómo validaste la propuesta?"
    ]
    assert split_speech_chunks("   ") == []


# ---------------------------------------------------------------------------
# S19 priority: live turns jump ahead of prefetch/filler synthesis
# ---------------------------------------------------------------------------


def test_priority_runner_runs_high_before_queued_low() -> None:
    runner = PrioritySerialRunner("test-priority")
    order: list[str] = []
    running = threading.Event()
    gate = threading.Event()

    def job(name: str, wait: bool = False):
        def work():
            order.append(name)
            if wait:
                running.set()
                gate.wait(5)
            return name

        return work

    first = runner.submit("low-1", job("low-1", wait=True), PRIORITY_LOW)
    assert running.wait(5)
    second = runner.submit("low-2", job("low-2"), PRIORITY_LOW)
    high = runner.submit("high", job("high"), PRIORITY_HIGH)

    gate.set()

    assert first.result(5) == "low-1"
    assert high.result(5) == "high"
    assert second.result(5) == "low-2"
    assert order == ["low-1", "high", "low-2"]


def test_priority_runner_promotes_a_pending_job_without_duplicating_it() -> None:
    runner = PrioritySerialRunner("test-promote")
    order: list[str] = []
    running = threading.Event()
    gate = threading.Event()
    calls: list[str] = []

    def blocker():
        order.append("blocker")
        running.set()
        gate.wait(5)
        return "blocker"

    def chunk():
        calls.append("chunk")
        order.append("chunk")
        return "chunk"

    first = runner.submit("blocker", blocker, PRIORITY_LOW)
    assert running.wait(5)

    low = runner.submit("chunk-0", chunk, PRIORITY_LOW)
    high = runner.submit("chunk-0", chunk, PRIORITY_HIGH)
    assert low is high

    other = runner.submit("other", lambda: order.append("other"), PRIORITY_LOW)

    gate.set()

    assert first.result(5) == "blocker"
    assert high.result(5) == "chunk"
    assert other.result(5) is None
    assert calls == ["chunk"]
    assert order == ["blocker", "chunk", "other"]


def test_synthesize_to_reuses_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "already.wav"
    target.write_bytes(b"RIFF-existing")

    result = synthesize_to("hola", "tutor", tmp_path, target)

    assert result == target
    assert target.read_bytes() == b"RIFF-existing"


def test_split_speech_chunks_caps_a_long_opening_sentence() -> None:
    text = (
        "En tu propuesta planteas una arquitectura en tres capas con lógica de "
        "negocio separada de la interfaz: ¿por qué era importante para esta "
        "empresa?"
    )

    chunks = split_speech_chunks(text)

    assert len(chunks[0]) <= 100
    assert " ".join(chunks) == " ".join(text.split())
    assert len(chunks) >= 2
