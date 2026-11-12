"""Integration wiring: config.json is the single source for voices and judges."""

import json
from pathlib import Path

from app import session, tts

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())


def test_voices_come_from_config(monkeypatch):
    monkeypatch.setattr(tts, "VOICES", dict(tts.VOICES))
    tts.configure_voices(CONFIG["roles"])
    for role, settings in CONFIG["roles"].items():
        assert tts.voice_for(role) == settings["voice"]


def test_config_voices_are_spanish_and_distinct_per_judge():
    judges = [CONFIG["roles"][r]["voice"] for r in session.JUDGE_ROLES]
    assert len(set(judges)) == len(judges)
    for voice in judges:
        assert "Inglés" not in voice


def test_judges_come_from_config(monkeypatch):
    monkeypatch.setattr(session, "JUDGES", session.JUDGES)
    session.configure_judges(CONFIG["roles"])
    names = [j.name for j in session._default_judges()]
    assert names == [CONFIG["roles"][r]["name"] for r in session.JUDGE_ROLES]
