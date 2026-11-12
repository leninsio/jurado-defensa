"""Tests for the FastAPI endpoints, using fakes for brain, STT and TTS."""

import asyncio
import base64
import json
import logging
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

# The test command runs from the repository root without a root conftest.py,
# so make the `app` package importable from this file's location.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import (  # noqa: E402
    chapter_scope_block,
    create_app,
    deliberation_prompt,
    direct_first_question_prompt,
    extract_json,
    first_question_prompt,
    follow_up_prompt,
    interjection_prompt,
    next_main_question_prompt,
    render_opening_line,
    room_history_block,
)
from app.session import JuryTurn, PracticeSession, SlideState  # noqa: E402
from app.tts import PRIORITY_HIGH, PRIORITY_LOW  # noqa: E402

# Long enough to pass the default jury minimum (80 content words), so the
# existing flow tests exercise a realistic exposition transcript.
EXPOSITION_TEXT = (
    "Buenas tardes, mi nombre es Luis y voy a presentar el proyecto de grado "
    "sobre un sistema de información para la gestión de inventario y el control "
    "de préstamos prendarios en Inversiones Ejemplo C.A. Primero "
    "explico el problema: los registros manuales retrasan las operaciones y "
    "dificultan el control de los préstamos. Luego presento el objetivo general, "
    "los objetivos específicos y la justificación del proyecto. Después describo "
    "la metodología, el tipo de investigación, la población, el censo y el "
    "instrumento de recolección de datos. Finalmente muestro los resultados "
    "esperados, las conclusiones y las recomendaciones para mejorar la gestión "
    "de inventario y préstamos de la empresa."
)


class FakeBrain:
    """Deterministic in-memory stand-in for app/brain.py."""

    def __init__(self):
        self.sessions = {}
        self.calls = []
        self.files = []
        self.counters = {}
        self.evaluator_reply = None
        self.fail_roles = set()
        self.deliberation_replies = {}
        self.judge_reply = None
        # Deadline tests: asks block until interrupt() releases their gate.
        self.block_counts: dict[str, int] = {}
        self.block_remaining = 0
        self.gates: dict[str, threading.Event] = {}
        self.interrupts: list[str] = []

    def block_asks(self, role=None, times=1):
        """Make the next ``times`` asks block (all roles when ``role`` is None)."""
        self.gates.setdefault(role or "*", threading.Event()).clear()
        if role is None:
            self.block_remaining = times
        else:
            self.block_counts[role] = times

    def interrupt(self, session_id):
        self.interrupts.append(session_id)
        for gate in self.gates.values():
            gate.set()

    def create_session(self, role):
        count = sum(1 for known in self.sessions.values() if known == role) + 1
        session_id = f"{role}-{count}"
        self.sessions[session_id] = role
        return session_id

    def ask(self, session_id, text, files=None):
        role = self.sessions[session_id]
        self.files.append((role, list(files or [])))
        if role in self.fail_roles:
            raise RuntimeError("brain down")
        blocked = False
        gate = None
        if self.block_remaining > 0:
            self.block_remaining -= 1
            blocked = True
            gate = self.gates.setdefault("*", threading.Event())
        elif self.block_counts.get(role, 0) > 0:
            self.block_counts[role] -= 1
            blocked = True
            gate = self.gates.setdefault(role, threading.Event())
        if blocked:
            # Wait for the server-side deadline to interrupt this turn.
            gate.clear()
            if not gate.wait(2.0):
                raise RuntimeError("blocked ask was never interrupted")
            raise RuntimeError("brain deadline reached")
        self.calls.append((session_id, role, text))
        count = self.counters.get(session_id, 0) + 1
        self.counters[session_id] = count
        if role.startswith("juez"):
            if "DELIBERACIÓN FINAL" in text:
                if role in self.deliberation_replies:
                    return self.deliberation_replies[role]
                exposicion = (
                    None if "El estudiante no realizó la exposición" in text else 18
                )
                return json.dumps(
                    {
                        "puntaje_exposicion": exposicion,
                        "puntaje_respuestas": 16,
                        "fortalezas": [
                            "Claridad al explicar",
                            "Buen manejo del tiempo",
                        ],
                        "debilidades": [
                            "Falta profundizar en la justificación"
                        ],
                        "comentario": f"Evaluación de {role}",
                    }
                )
            if self.judge_reply:
                return self.judge_reply
            return f"Pregunta {count} de {role}"
        if role == "evaluador":
            if self.evaluator_reply is not None:
                return self.evaluator_reply
            return json.dumps(
                {
                    "slide": 1,
                    "fortalezas": ["Claridad"],
                    "correcciones": ["Profundiza en la justificación"],
                    "frase_sugerida": "Como se observa en la figura...",
                }
            )
        if role == "tutor":
            return f"Tutor responde: {text}"
        return "ok"


class FakeSTT:
    """Stand-in for app/stt.py."""

    def __init__(self, text=EXPOSITION_TEXT):
        self.text = text
        self.error = None
        self.paths = []
        self.segments = []
        self.segment_error = None

    def transcribe_audio(self, path):
        self.paths.append(str(path))
        if self.error is not None:
            raise self.error
        return self.text

    def transcribe_segments(self, path):
        self.paths.append(str(path))
        if self.segment_error is not None:
            raise self.segment_error
        return list(self.segments)


class FakeTTS:
    """Stand-in for app/tts.py with control over when chunk audio exists.

    ``hold`` keeps requested chunks pending until ``release()`` runs, which lets
    a test prove that the question text reaches the client before its audio.
    """

    def __init__(self):
        self.calls = []
        self.requests = []
        self.pending = {}
        self.priorities = {}
        self.hold = False

    def synthesize(self, text, role, out_dir, priority=0):
        self.calls.append((text, role))
        path = Path(out_dir) / f"{role}-{len(self.calls)}.wav"
        path.write_bytes(b"RIFF....WAVEfake")
        return path

    def request_synthesis(self, text, role, out_dir, target, priority=0):
        target = Path(target)
        if target.is_file() and target.stat().st_size > 0:
            ready = Future()
            ready.set_result(target)
            return ready
        self.requests.append((text, role, priority))
        entry = self.pending.get(target.name)
        if entry is not None:
            future, previous_priority, held_text, held_role, held_target = entry
            final_priority = min(previous_priority, priority)
            self.pending[target.name] = (
                future,
                final_priority,
                held_text,
                held_role,
                held_target,
            )
            self.priorities[target.name] = final_priority
            return future
        future = Future()
        self.pending[target.name] = (future, priority, text, role, target)
        self.priorities[target.name] = priority
        if not self.hold:
            self._complete(target.name)
        return future

    def _complete(self, name):
        future, _priority, text, role, target = self.pending.pop(name)
        self.calls.append((text, role))
        Path(target).write_bytes(b"RIFF....WAVEfake")
        if not future.done():
            future.set_result(target)

    def release(self):
        for name in list(self.pending):
            self._complete(name)

    def synthesize_to(self, text, role, out_dir, target, priority=0):
        return self.request_synthesis(
            text, role, out_dir, target, priority=priority
        ).result()


class RecordingExecutor:
    """Queue background work so a test decides exactly when it runs."""

    def __init__(self):
        self.tasks = []
        self.shutdown_called = False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        self.tasks.append((future, fn, args, kwargs))
        return future

    def run_pending(self):
        while self.tasks:
            future, fn, args, kwargs = self.tasks.pop(0)
            if future.done():
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 - the future carries it
                future.set_exception(exc)

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_called = True
        self.tasks.clear()


@pytest.fixture
def slides_dir(tmp_path):
    directory = tmp_path / "slides"
    directory.mkdir()
    for number in (1, 2, 3):
        (directory / f"slide-{number:02d}.png").write_bytes(
            b"\x89PNG\r\n\x1a\nfake"
        )
        (directory / f"slide-{number:02d}.txt").write_text(
            f"Texto de la lámina {number}", encoding="utf-8"
        )
    return directory


@pytest.fixture
def brain():
    return FakeBrain()


@pytest.fixture
def stt():
    return FakeSTT()


@pytest.fixture
def tts():
    return FakeTTS()


@pytest.fixture
def make_client(tmp_path, slides_dir, brain, stt, tts):
    """Build extra TestClients with different jury turn-taking settings."""
    clients = []

    def factory(**overrides):
        app = create_app(
            brain=brain,
            stt=stt,
            tts=tts,
            slides_dir=slides_dir,
            audio_dir=tmp_path / "audio",
            session_dir=tmp_path / "practicas",
            **overrides,
        )
        test_client = TestClient(app)
        test_client.__enter__()
        clients.append(test_client)
        return test_client

    yield factory
    for test_client in clients:
        test_client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    # Deterministic jury: no interjections and a seeded RNG for the asker order.
    yield make_client(rng=random.Random(0), interjection_probability=0.0)


def start_session(client):
    response = client.post("/api/session")
    assert response.status_code == 200, response.text
    return response.json()


def upload_slide(client, number, duration=10.0):
    return client.post(
        f"/api/exposition/{number}/audio",
        files={"file": ("slide.webm", b"fake-audio-bytes", "audio/webm")},
        data={"duration": str(duration)},
    )


def prepare_finished_exposition(client, slides=(1, 2)):
    start_session(client)
    for number in slides:
        assert upload_slide(client, number).status_code == 200
    response = client.post("/api/session/finish")
    assert response.status_code == 200, response.text
    return response.json()


def start_jury(client, questions_per_judge=2, chapters=None):
    payload = {"questions_per_judge": questions_per_judge}
    if chapters is not None:
        payload["chapters"] = chapters
    return client.post("/api/jury/start", json=payload)


def start_direct_jury(client, questions_per_judge=2, chapters=None):
    payload = {"questions_per_judge": questions_per_judge}
    if chapters is not None:
        payload["chapters"] = chapters
    return client.post("/api/jury/direct", json=payload)


def answer(client, text):
    # The client always echoes the turn it is answering (T11): the server
    # rejects a stale turn instead of recording the same answer twice.
    turn_id = current_jury(client).get("current_turn_id")
    return client.post(
        "/api/jury/answer", json={"text": text, "turn_id": turn_id}
    )


def current_jury(client):
    return client.get("/api/session").json()["jury"]


def wait_for_jury_done(client, limit=500):
    for _ in range(limit):
        jury = current_jury(client)
        if jury["status"] == "done":
            return jury
        time.sleep(0.01)
    raise AssertionError("The jury deliberation did not finish in time.")


def answer_until_done(client, limit=60):
    bodies = []
    for index in range(limit):
        response = answer(client, f"respuesta {index + 1}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["result"]["type"] == "deliberating":
            jury = wait_for_jury_done(client)
            bodies.append(
                {
                    "jury": jury,
                    "result": {
                        "type": "verdict",
                        "done": True,
                        "verdict": None,
                        "final": jury["final"],
                    },
                }
            )
            return bodies
        bodies.append(body)
        if body["result"]["done"]:
            return bodies
    raise AssertionError("The jury did not finish within the answer limit.")


def asker_sequence(make_client, seed, questions_per_judge=1):
    """Main-asker roles for one deterministic run with no interjections."""
    jury_client = make_client(
        rng=random.Random(seed), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=questions_per_judge)
    jury = current_jury(jury_client)
    roles = [jury["current_role"]]
    for _ in range(questions_per_judge * 3):
        body = answer(jury_client, "respuesta").json()
        if body["result"]["type"] == "deliberating":
            break
        roles.append(body["jury"]["current_role"])
    return roles


# ---------------------------------------------------------------------------
# Slides and session basics
# ---------------------------------------------------------------------------


def test_health_reports_slide_count(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "slides": 3}


def test_index_serves_the_single_page_ui(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "Simulador de jurado" in body
    assert "/static/styles.css" in body
    assert "/static/app.js" in body
    for tab in ("Exposición", "Jurado", "Retroalimentación", "Tutor"):
        assert tab in body


def test_static_assets_are_served(client):
    script = client.get("/static/app.js")
    styles = client.get("/static/styles.css")

    assert script.status_code == 200
    assert styles.status_code == 200
    assert "startMic" in script.text
    assert ".ptt" in styles.text


def test_slide_listing_is_sorted_and_paired(client):
    response = client.get("/api/slides")

    assert response.status_code == 200
    slides = response.json()["slides"]
    assert [slide["number"] for slide in slides] == [1, 2, 3]
    assert slides[0]["text"] == "Texto de la lámina 1"
    assert re.fullmatch(r"/api/slides/1/image\?v=\d+", slides[0]["image_url"])


def test_slide_image_is_served(client):
    response = client.get("/api/slides/2/image")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_unknown_slide_image_returns_404(client):
    assert client.get("/api/slides/99/image").status_code == 404


def test_session_is_created_and_listed(client):
    assert client.get("/api/session").status_code == 404

    created = start_session(client)

    assert created["id"]
    assert created["exposition_status"] == "running"
    assert created["current_slide"] == 1
    assert len(created["slides"]) == 3
    assert [judge["role"] for judge in created["jury"]["judges"]] == [
        "juez_metodologo",
        "juez_tecnico",
        "juez_presidente",
    ]
    assert all(judge["name"] and judge["persona"] for judge in created["jury"]["judges"])
    assert created["jury"]["current_role"] is None
    assert created["jury"]["turn_kind"] is None
    assert all(
        judge["remaining"] == created["jury"]["questions_per_judge"]
        for judge in created["jury"]["judges"]
    )

    current = client.get("/api/session").json()
    assert current["id"] == created["id"]


def test_exposition_start_records_start_time(client):
    start_session(client)

    response = client.post("/api/exposition/start")

    assert response.status_code == 200
    assert response.json()["elapsed_seconds"] >= 0


# ---------------------------------------------------------------------------
# Exposition audio upload and background transcription
# ---------------------------------------------------------------------------


def test_audio_upload_transcribes_and_stores_slide_state(client, stt):
    start_session(client)

    response = upload_slide(client, 1, duration=42.5)

    assert response.status_code == 200
    assert response.json()["slide"] == 1
    assert response.json()["transcription_status"] == "transcribing"
    assert len(stt.paths) == 1

    session = client.get("/api/session").json()
    slide = session["slides"][0]
    assert slide["transcript"] == EXPOSITION_TEXT
    assert slide["transcription_status"] == "done"
    assert slide["duration_seconds"] == 42.5
    assert slide["has_audio"] is True
    assert re.fullmatch(r"/api/slides/1/image\?v=\d+", slide["image_url"])


def test_audio_upload_empty_transcript_asks_to_repeat(client, stt):
    stt.text = "   "
    start_session(client)

    upload_slide(client, 1)

    slide = client.get("/api/session").json()["slides"][0]
    assert slide["transcript"] is None
    assert slide["transcription_status"] == "empty"


def test_audio_upload_stt_error_is_visible(client, stt):
    stt.error = RuntimeError("whisper boom")
    start_session(client)

    upload_slide(client, 1)

    slide = client.get("/api/session").json()["slides"][0]
    assert slide["transcription_status"] == "error"
    assert "whisper boom" in slide["transcription_error"]


def test_audio_upload_unknown_slide_returns_404(client):
    start_session(client)

    assert upload_slide(client, 99).status_code == 404


def test_finish_saves_session_json(client):
    payload = prepare_finished_exposition(client, slides=(1,))

    saved_path = Path(payload["saved_path"])
    assert saved_path.is_file()
    assert saved_path.parent.name == "practicas"

    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["id"] == payload["session"]["id"]
    assert saved["exposition_status"] == "finished"
    assert saved["slides"][0]["transcript"] == EXPOSITION_TEXT


# ---------------------------------------------------------------------------
# Real defense mode: opening, one continuous recording, slide timeline
# ---------------------------------------------------------------------------


def upload_full_audio(client, stt, timeline, segments, duration=30.0):
    stt.segments = segments
    return client.post(
        "/api/exposition/full/audio",
        files={"file": ("defensa.webm", b"fake-full-audio", "audio/webm")},
        data={"timeline": json.dumps(timeline), "duration": str(duration)},
    )


def test_render_opening_line_uses_the_student_name():
    assert (
        render_opening_line("Buenas tardes. Adelante, {name}, ya puedes empezar.", "Ana")
        == "Buenas tardes. Adelante, Ana, ya puedes empezar."
    )
    assert render_opening_line("", "Ana").startswith("Buenas tardes.")


def test_config_declares_real_defense_settings():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "config.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["student"]["name"] == "Tu Nombre"
    assert (
        config["jury"]["opening_line"]
        == "Buenas tardes. Adelante, {name}, ya puedes empezar."
    )
    assert config["jury"]["target_minutes_min"] == 15
    assert config["jury"]["target_minutes_max"] == 18
    assert config["jury"]["interjection_probability"] == 0.35


def test_opening_line_is_spoken_by_the_president_judge(client, tts):
    start_session(client)

    response = client.post("/api/exposition/opening")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "Buenas tardes. Adelante, Tu Nombre, ya puedes empezar."
    assert body["audio_url"].startswith("/api/audio/")
    assert tts.calls[-1] == (body["text"], "juez_presidente")
    assert client.get(body["audio_url"]).status_code == 200


def test_exposition_start_accepts_the_real_mode(client):
    start_session(client)

    response = client.post("/api/exposition/start", json={"mode": "real"})

    assert response.status_code == 200, response.text
    assert response.json()["exposition_mode"] == "real"
    assert client.post(
        "/api/exposition/start", json={"mode": "inventado"}
    ).status_code == 422


def test_session_exposes_targets_and_defaults_to_practice_mode(client):
    session = start_session(client)

    assert session["exposition_mode"] == "practice"
    assert session["slide_timeline"] == []
    assert session["targets"] == {"min_minutes": 15, "max_minutes": 18}


def test_full_audio_assigns_segments_to_slides_by_midpoint(client, stt):
    start_session(client)

    response = upload_full_audio(
        client,
        stt,
        [
            {"slide": 1, "start_ms": 0},
            {"slide": 2, "start_ms": 10000},
            {"slide": 3, "start_ms": 20000},
        ],
        [
            (0.0, 4.0, "introducción"),
            # Starts on slide 1 but its midpoint (10.5 s) belongs to slide 2.
            (9.0, 12.0, "metodología"),
            (21.0, 24.0, "cierre"),
        ],
    )

    assert response.status_code == 200, response.text
    session = client.get("/api/session").json()
    assert session["exposition_mode"] == "real"
    assert session["slide_timeline"] == [
        {"slide": 1, "start_ms": 0},
        {"slide": 2, "start_ms": 10000},
        {"slide": 3, "start_ms": 20000},
    ]
    slides = {slide["number"]: slide for slide in session["slides"]}
    assert slides[1]["transcript"] == "introducción"
    assert slides[2]["transcript"] == "metodología"
    assert slides[3]["transcript"] == "cierre"
    assert [slides[number]["duration_seconds"] for number in (1, 2, 3)] == [
        10.0,
        10.0,
        10.0,
    ]
    assert all(
        slides[number]["transcription_status"] == "done" for number in (1, 2, 3)
    )
    assert all(slides[number]["has_audio"] for number in (1, 2, 3))


def test_full_audio_accumulates_a_revisited_slide(client, stt):
    start_session(client)

    response = upload_full_audio(
        client,
        stt,
        [
            {"slide": 1, "start_ms": 0},
            {"slide": 2, "start_ms": 10000},
            {"slide": 1, "start_ms": 20000},
        ],
        [
            (0.0, 4.0, "primera"),
            (11.0, 14.0, "segunda"),
            (21.0, 24.0, "tercera"),
        ],
    )

    assert response.status_code == 200, response.text
    slides = {
        slide["number"]: slide
        for slide in client.get("/api/session").json()["slides"]
    }
    assert slides[1]["transcript"] == "primera tercera"
    assert slides[1]["duration_seconds"] == 20.0
    assert slides[2]["transcript"] == "segunda"
    assert slides[2]["duration_seconds"] == 10.0


def test_full_audio_keeps_silent_visited_slides_visible(client, stt):
    start_session(client)

    response = upload_full_audio(
        client,
        stt,
        [
            {"slide": 1, "start_ms": 0},
            {"slide": 2, "start_ms": 10000},
            {"slide": 3, "start_ms": 20000},
        ],
        [(0.0, 4.0, "inicio"), (21.0, 24.0, "final")],
    )

    assert response.status_code == 200, response.text
    slides = {
        slide["number"]: slide
        for slide in client.get("/api/session").json()["slides"]
    }
    assert slides[1]["transcription_status"] == "done"
    assert slides[2]["transcription_status"] == "empty"
    assert slides[2]["transcript"] is None
    assert slides[2]["duration_seconds"] == 10.0
    assert slides[3]["transcription_status"] == "done"
    assert slides[3]["has_audio"] is True


def test_full_audio_rejects_an_invalid_timeline(client, stt):
    start_session(client)

    empty = upload_full_audio(client, stt, [], [])
    invalid_json = client.post(
        "/api/exposition/full/audio",
        files={"file": ("defensa.webm", b"fake-full-audio", "audio/webm")},
        data={"timeline": "no es json"},
    )
    unknown_slide = upload_full_audio(
        client, stt, [{"slide": 99, "start_ms": 0}], []
    )

    assert empty.status_code == 422
    assert invalid_json.status_code == 422
    assert unknown_slide.status_code == 422


def test_full_audio_stt_error_is_visible(client, stt):
    start_session(client)
    stt.segment_error = RuntimeError("segment boom")

    response = upload_full_audio(
        client,
        stt,
        [{"slide": 1, "start_ms": 0}],
        [],
    )

    assert response.status_code == 200, response.text
    slides = client.get("/api/session").json()["slides"]
    assert slides[0]["transcription_status"] == "error"
    assert "segment boom" in slides[0]["transcription_error"]


def test_per_slide_practice_mode_still_works(client, stt):
    session = start_session(client)
    assert session["exposition_mode"] == "practice"

    upload_slide(client, 1, duration=42.5)

    current = client.get("/api/session").json()
    assert current["exposition_mode"] == "practice"
    assert current["slide_timeline"] == []
    assert current["slides"][0]["transcript"] == EXPOSITION_TEXT
    assert current["slides"][0]["duration_seconds"] == 42.5
    assert current["slides"][1]["has_audio"] is False


# ---------------------------------------------------------------------------
# Jury flow
# ---------------------------------------------------------------------------


def test_jury_requires_finished_exposition(client):
    start_session(client)

    response = start_jury(client)

    assert response.status_code == 409


def test_jury_rejects_filler_only_exposition(client, stt, brain):
    stt.text = "Ok, ok"
    prepare_finished_exposition(client, slides=(1,))

    response = start_jury(client)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "demasiado breve" in detail
    assert "0 palabras" in detail
    assert "al menos 80" in detail
    assert "Expón tus láminas" in detail
    assert not any(role.startswith("juez") for role in brain.sessions.values())


def test_jury_accepts_exposition_at_the_word_threshold(client, stt):
    stt.text = " ".join(["contenido"] * 80)
    prepare_finished_exposition(client, slides=(1,))

    response = start_jury(client)

    assert response.status_code == 200, response.text


def test_jury_rejects_exposition_one_word_below_the_threshold(client, stt):
    stt.text = " ".join(["contenido"] * 79) + " ok, eh, mm"
    prepare_finished_exposition(client, slides=(1,))

    response = start_jury(client)

    assert response.status_code == 409
    assert "79 palabras" in response.json()["detail"]


def test_jury_full_turn_flow(client, brain, tts):
    prepare_finished_exposition(client)

    response = start_jury(client, questions_per_judge=2)
    assert response.status_code == 200, response.text
    body = response.json()
    jury = body["jury"]
    judge_roles = {judge["role"] for judge in jury["judges"]}
    assert jury["turn_kind"] == "pregunta"
    assert jury["current_role"] in judge_roles
    assert jury["current_audio_url"].startswith("/api/audio/")

    bodies = answer_until_done(client)
    final = bodies[-1]
    assert final["result"]["type"] == "verdict"
    assert final["result"]["done"] is True
    assert final["result"]["final"]["average"] == 17.0
    assert final["result"]["final"]["final_grade"] == 17.0
    assert len(final["result"]["final"]["verdicts"]) == 3
    assert final["jury"]["status"] == "done"
    assert final["jury"]["current_question"] is None
    assert final["jury"]["current_role"] is None
    assert final["jury"]["turn_kind"] is None
    assert all(judge["asked"] == 2 for judge in final["jury"]["judges"])
    assert all(judge["remaining"] == 0 for judge in final["jury"]["judges"])
    assert all(judge["verdict"] for judge in final["jury"]["judges"])

    # The full exposition transcript is sent once per judge, never again; the
    # final deliberation runs once and also covers the whole defense.
    for role in ("juez_metodologo", "juez_tecnico", "juez_presidente"):
        prompts = [call[2] for call in brain.calls if call[1] == role]
        assert (
            sum(
                "Transcripción completa de la exposición" in prompt
                for prompt in prompts
            )
            == 1
        )
        assert sum("DELIBERACIÓN FINAL" in prompt for prompt in prompts) == 1

    # Six main questions plus one follow-up per judge, plus the closing.
    assert len(tts.calls) == 10
    assert any(
        voice == "juez_presidente" and text.startswith("El jurado")
        for text, voice in tts.calls
    )


def test_jury_picks_a_random_next_asker_and_never_repeats_in_a_row(make_client):
    sequences = [tuple(asker_sequence(make_client, seed)) for seed in (7, 7, 4)]

    first = sequences[0]
    assert len(first) == 3
    assert set(first) == {
        "juez_metodologo",
        "juez_tecnico",
        "juez_presidente",
    }
    assert all(left != right for left, right in zip(first, first[1:]))
    assert sequences[0] == sequences[1]
    assert len(set(sequences)) > 1
    assert first != ("juez_metodologo", "juez_tecnico", "juez_presidente")


def test_jury_gives_each_judge_exactly_its_main_budget(make_client):
    jury_client = make_client(
        rng=random.Random(5), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=2)

    bodies = answer_until_done(jury_client)
    jury = bodies[-1]["jury"]

    assert jury["status"] == "done"
    assert [judge["asked"] for judge in jury["judges"]] == [2, 2, 2]
    assert [judge["remaining"] for judge in jury["judges"]] == [0, 0, 0]
    main_turns = [
        turn
        for judge in jury["judges"]
        for turn in judge["turns"]
        if turn["kind"] == "pregunta"
    ]
    assert len(main_turns) == 6


def test_jury_interjects_from_another_judge_at_probability_one(
    make_client, brain
):
    jury_client = make_client(
        rng=random.Random(3), interjection_probability=1.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)
    first = current_jury(jury_client)
    first_role = first["current_role"]
    first_question = first["current_question"]

    body = answer(jury_client, "el censo son tres trabajadores").json()
    jury = body["jury"]

    assert jury["turn_kind"] == "intervencion"
    assert jury["current_role"] != first_role
    assert jury["current_question"]
    prompts = [call[2] for call in brain.calls if call[1] == jury["current_role"]]
    prompt = prompts[-1]
    assert "colega" in prompt
    assert first_question in prompt
    assert "el censo son tres trabajadores" in prompt


def test_jury_sends_no_interjection_at_probability_zero(make_client):
    jury_client = make_client(
        rng=random.Random(3), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)

    body = answer(jury_client, "respuesta").json()
    jury = body["jury"]

    assert jury["turn_kind"] == "pregunta"
    assert all(
        turn["kind"] != "intervencion"
        for judge in jury["judges"]
        for turn in judge["turns"]
    )


def test_jury_interjection_does_not_consume_main_budget(make_client):
    jury_client = make_client(
        rng=random.Random(3), interjection_probability=1.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)

    jury = answer(jury_client, "respuesta").json()["jury"]
    interjector = next(
        judge for judge in jury["judges"] if judge["role"] == jury["current_role"]
    )
    assert interjector["asked"] == 0
    assert interjector["interjections"] == 1
    assert interjector["remaining"] == 1

    bodies = answer_until_done(jury_client)
    final = bodies[-1]["jury"]
    assert sum(judge["asked"] for judge in final["judges"]) == 3
    assert all(judge["remaining"] == 0 for judge in final["judges"])


def test_jury_emits_all_verdicts_after_the_budgets(make_client):
    jury_client = make_client(
        rng=random.Random(4), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)

    first = answer(jury_client, "uno").json()
    assert first["result"]["type"] == "question"
    assert all(judge["verdict"] is None for judge in first["jury"]["judges"])

    second = answer(jury_client, "dos").json()
    assert second["result"]["type"] == "question"
    assert all(judge["verdict"] is None for judge in second["jury"]["judges"])

    third = answer(jury_client, "tres").json()
    assert third["result"]["type"] == "deliberating"
    assert third["result"]["done"] is False
    assert third["result"]["final"] is None
    assert third["jury"]["status"] == "deliberating"
    assert third["jury"]["current_question"] is None

    jury = wait_for_jury_done(jury_client)
    final = jury["final"]
    assert final["average"] == 17.0
    assert final["final_grade"] == 17.0
    assert final["result"] == "Aprobado"
    assert len(final["verdicts"]) == 3
    assert all(judge["verdict"] for judge in jury["judges"])
    assert all(judge["score"] == 17.0 for judge in jury["judges"])
    assert jury["current_question"] is None
    assert jury["current_role"] is None
    assert jury["turn_kind"] is None


def test_jury_keeps_one_follow_up_for_the_original_asker(make_client):
    jury_client = make_client(
        rng=random.Random(2), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=2)
    first_role = current_jury(jury_client)["current_role"]

    body = answer(jury_client, "respuesta principal").json()
    jury = body["jury"]
    assert jury["turn_kind"] == "repregunta"
    assert jury["current_role"] == first_role

    body = answer(jury_client, "respuesta a la repregunta").json()
    assert body["jury"]["turn_kind"] == "pregunta"

    final = answer_until_done(jury_client, limit=60)[-1]["jury"]
    assert [judge["follow_ups"] for judge in final["judges"]] == [1, 1, 1]
    follow_up_turns = [
        turn
        for judge in final["judges"]
        for turn in judge["turns"]
        if turn["kind"] == "repregunta"
    ]
    assert len(follow_up_turns) == 3


def test_jury_mixed_flow_keeps_the_exact_budgets(make_client):
    jury_client = make_client(
        rng=random.Random(9), interjection_probability=0.35
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=2)

    bodies = answer_until_done(jury_client, limit=80)
    final = bodies[-1]
    jury = final["jury"]

    assert final["result"]["done"] is True
    assert jury["status"] == "done"
    assert [judge["asked"] for judge in jury["judges"]] == [2, 2, 2]
    assert [judge["remaining"] for judge in jury["judges"]] == [0, 0, 0]
    assert all(judge["follow_ups"] == 1 for judge in jury["judges"])
    assert sum(judge["interjections"] for judge in jury["judges"]) >= 1
    assert len(final["result"]["final"]["verdicts"]) == 3


def _async_jury_app(tmp_path, slides_dir, brain, stt, tts, executor):
    return create_app(
        brain=brain,
        stt=stt,
        tts=tts,
        slides_dir=slides_dir,
        audio_dir=tmp_path / "audio",
        session_dir=tmp_path / "practicas",
        rng=random.Random(0),
        interjection_probability=0.0,
        executor=executor,
        deliberation_executor=executor,
    )


def test_concurrent_identical_answers_are_applied_once(
    tmp_path, slides_dir, brain, stt, tts
):
    """Two identical answers for the same turn never duplicate the exchange (T11)."""
    background = ThreadPoolExecutor(max_workers=6)
    app = _async_jury_app(
        tmp_path, slides_dir, brain, stt, tts, background
    )

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            assert (await client.post("/api/session")).status_code == 200
            started = await client.post(
                "/api/jury/direct", json={"questions_per_judge": 2}
            )
            assert started.status_code == 200, started.text
            jury = (await client.get("/api/session")).json()["jury"]
            turn_id = jury["current_turn_id"]
            assert turn_id is not None
            payload = {"text": "respuesta duplicada", "turn_id": turn_id}
            responses = await asyncio.gather(
                client.post("/api/jury/answer", json=payload),
                client.post("/api/jury/answer", json=payload),
            )
            statuses = sorted(response.status_code for response in responses)
            jury = (await client.get("/api/session")).json()["jury"]
            index = 0
            while jury["status"] == "awaiting_answer" and index < 20:
                index += 1
                body = await client.post(
                    "/api/jury/answer",
                    json={
                        "text": f"respuesta {index + 1}",
                        "turn_id": jury["current_turn_id"],
                    },
                )
                assert body.status_code == 200, body.text
                jury = (await client.get("/api/session")).json()["jury"]
            for _ in range(300):
                if jury["status"] != "deliberating":
                    break
                await asyncio.sleep(0.01)
                jury = (await client.get("/api/session")).json()["jury"]
            return statuses, responses, jury

    try:
        statuses, responses, jury = asyncio.run(scenario())
    finally:
        background.shutdown(wait=False, cancel_futures=True)

    assert statuses == [200, 409]
    rejected = next(
        response for response in responses if response.status_code == 409
    )
    assert rejected.json()["detail"] == "Esa pregunta ya fue respondida."

    assert jury["status"] == "done"
    assert [judge["asked"] for judge in jury["judges"]] == [2, 2, 2]
    assert [judge["follow_ups"] for judge in jury["judges"]] == [1, 1, 1]
    assert all(
        judge["interjections"] == 0 for judge in jury["judges"]
    )
    turns = [
        turn for judge in jury["judges"] for turn in judge["turns"]
    ]
    assert len(turns) == 9
    assert sum(1 for turn in turns if turn["kind"] == "pregunta") == 6
    assert sum(1 for turn in turns if turn["kind"] == "repregunta") == 3
    assert (
        sum(1 for turn in turns if turn["answer"] == "respuesta duplicada")
        == 1
    )


def test_stale_turn_id_is_rejected_without_state_change(client):
    prepare_finished_exposition(client)
    start_jury(client)
    stale_turn_id = current_jury(client)["current_turn_id"]
    assert answer(client, "respuesta original").status_code == 200
    before = current_jury(client)

    response = client.post(
        "/api/jury/answer",
        json={"text": "otra respuesta", "turn_id": stale_turn_id},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Esa pregunta ya fue respondida."
    assert current_jury(client) == before


def test_answer_without_turn_id_is_rejected(client):
    prepare_finished_exposition(client)
    start_jury(client)
    before = current_jury(client)

    response = client.post("/api/jury/answer", json={"text": "sin turno"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Esa pregunta ya fue respondida."
    assert current_jury(client) == before


def test_identical_answer_for_the_same_turn_is_rejected(client, brain):
    prepare_finished_exposition(client)
    start_jury(client)
    jury = current_jury(client)
    turn_id = jury["current_turn_id"]
    role = jury["current_role"]
    # The follow-up generation fails after the answer was recorded, so the same
    # turn stays pending: an identical retry must not record a duplicate turn.
    brain.fail_roles.add(role)

    first = answer(client, "respuesta repetida")

    assert first.status_code == 502
    pending = current_jury(client)
    assert pending["status"] == "awaiting_answer"
    assert pending["current_turn_id"] == turn_id
    assert pending["current_question"] == jury["current_question"]

    second = client.post(
        "/api/jury/answer",
        json={"text": "respuesta repetida", "turn_id": turn_id},
    )

    assert second.status_code == 409
    assert second.json()["detail"] == "Esa pregunta ya fue respondida."
    assert current_jury(client) == pending


def test_jury_questions_per_judge_one(client):
    prepare_finished_exposition(client)
    start_jury(client, questions_per_judge=1)
    for _ in range(3):
        body = answer(client, "respuesta única").json()

    assert body["result"]["type"] == "deliberating"
    jury = wait_for_jury_done(client)
    assert jury["status"] == "done"
    assert all(judge["asked"] == 1 for judge in jury["judges"])
    assert all(len(judge["turns"]) == 1 for judge in jury["judges"])
    assert len(jury["final"]["verdicts"]) == 3


def test_jury_rejects_invalid_question_count(client):
    prepare_finished_exposition(client)

    response = start_jury(client, questions_per_judge=5)

    assert response.status_code == 422


def test_jury_answer_before_start_returns_409(client):
    prepare_finished_exposition(client)

    assert answer(client, "hola").status_code == 409


def test_jury_audio_answer_uses_transcript(client, stt):
    prepare_finished_exposition(client)
    start_jury(client)
    first_role = current_jury(client)["current_role"]
    stt.text = "respuesta hablada"

    response = client.post(
        "/api/jury/answer/audio",
        files={"file": ("answer.webm", b"fake-answer", "audio/webm")},
        data={"turn_id": str(current_jury(client)["current_turn_id"])},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    asked = next(
        judge
        for judge in body["jury"]["judges"]
        if judge["role"] == first_role
    )
    assert asked["turns"][0]["answer"] == "respuesta hablada"


def test_jury_empty_answer_does_not_advance(client, stt):
    prepare_finished_exposition(client)
    start_jury(client)
    first_role = current_jury(client)["current_role"]
    first_question = current_jury(client)["current_question"]
    stt.text = ""

    response = client.post(
        "/api/jury/answer/audio",
        files={"file": ("answer.webm", b"fake-answer", "audio/webm")},
        data={"turn_id": str(current_jury(client)["current_turn_id"])},
    )

    assert response.status_code == 422
    assert "repite" in response.json()["detail"].lower()
    jury = current_jury(client)
    asked = next(judge for judge in jury["judges"] if judge["role"] == first_role)
    assert asked["asked"] == 1
    assert asked["turns"] == []
    assert jury["current_question"] == first_question


def test_jury_brain_error_is_visible(client, brain):
    brain.fail_roles = {
        "juez_metodologo",
        "juez_tecnico",
        "juez_presidente",
    }
    prepare_finished_exposition(client)

    response = start_jury(client)

    assert response.status_code == 502
    assert "juez" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Evaluator report
# ---------------------------------------------------------------------------


def test_evaluator_parses_fenced_json_with_surrounding_text(client, brain):
    start_session(client)
    upload_slide(client, 1)
    brain.evaluator_reply = (
        "Claro, aquí está:\n"
        "```json\n"
        '{"slide": 1, "fortalezas": ["Claridad", "Buen ritmo"], '
        '"correcciones": ["Falta cita"], '
        '"frase_sugerida": "Como se observa en la figura..."}\n'
        "```\n"
        "Saludos."
    )

    response = client.post("/api/feedback/1")

    assert response.status_code == 200, response.text
    feedback = response.json()["feedback"]
    assert feedback["parse_error"] is False
    assert feedback["fortalezas"] == ["Claridad", "Buen ritmo"]
    assert feedback["correcciones"] == ["Falta cita"]
    assert feedback["frase_sugerida"] == "Como se observa en la figura..."

    report = client.get("/api/feedback").json()["report"]
    assert report["slides"][0]["status"] == "ok"
    assert report["slides"][0]["fortalezas"] == ["Claridad", "Buen ritmo"]


def test_evaluator_malformed_json_is_defensive(client, brain):
    start_session(client)
    upload_slide(client, 1)
    brain.evaluator_reply = "No pude generar la respuesta pedida."

    response = client.post("/api/feedback/1")

    assert response.status_code == 200, response.text
    feedback = response.json()["feedback"]
    assert feedback["parse_error"] is True
    assert feedback["fortalezas"] == []
    assert feedback["correcciones"] == []
    assert feedback["frase_sugerida"] == ""
    assert "No pude generar" in feedback["raw"]


def test_feedback_requires_a_transcript(client):
    start_session(client)

    response = client.post("/api/feedback/1")

    assert response.status_code == 409


def test_feedback_rejects_a_filler_only_slide(client, stt):
    stt.text = "Ok, ok, eh, mm"
    start_session(client)
    upload_slide(client, 1)

    response = client.post("/api/feedback/1")

    assert response.status_code == 409
    assert "Sin exposición suficiente en esta lámina" in response.json()["detail"]


def test_feedback_all_skips_filler_only_slides(client, stt, brain):
    start_session(client)
    upload_slide(client, 1)
    stt.text = "Ok, ok, eh, mm"
    upload_slide(client, 2)

    response = client.post("/api/feedback")

    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["slides"][0]["status"] == "ok"
    assert report["slides"][1]["status"] == "insufficient"
    assert "Sin exposición suficiente en esta lámina" in report["slides"][1]["message"]
    evaluator_calls = [call for call in brain.calls if call[1] == "evaluador"]
    assert len(evaluator_calls) == 1


def test_feedback_all_evaluates_slides_with_transcripts(client):
    start_session(client)
    upload_slide(client, 1, duration=30)
    upload_slide(client, 2, duration=45)

    response = client.post("/api/feedback")

    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["totals"] == {
        "slides": 3,
        "evaluated": 2,
        "duration_seconds": 75.0,
        "duration_label": "01:15",
    }
    assert report["slides"][0]["status"] == "ok"
    assert report["slides"][2]["status"] == "no_transcript"


def test_feedback_markdown_export(client):
    start_session(client)
    upload_slide(client, 1, duration=30)
    client.post("/api/feedback/1")

    response = client.get("/api/feedback/markdown")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "attachment" in response.headers["content-disposition"]
    body = response.text
    assert body.startswith("# Retroalimentación por lámina")
    assert "## Lámina 1 · 00:30" in body
    assert "**Fortalezas**" in body
    assert "Como se observa en la figura..." in body


def test_feedback_adds_written_fields_and_python_metrics(client, brain, stt):
    start_session(client)
    stt.text = "eh eh este ok proyecto inventario"
    upload_slide(client, 1, duration=60)
    brain.evaluator_reply = json.dumps(
        {
            "slide": 1,
            "dijo": "Explicó el inventario del proyecto.",
            "fortalezas": ["Claridad"],
            "correcciones": ["Falta cita"],
            "omisiones": ["No mencionó la población"],
            "errores_vs_tesis": ["Dijo que la muestra era el censo"],
            "frase_sugerida": "El censo incluye a los cinco trabajadores.",
            "puntaje": 7.5,
            # The model must not own these metrics; Python computes them.
            "muletillas": 99,
            "palabras_por_minuto": 999,
        }
    )

    response = client.post("/api/feedback/1")

    assert response.status_code == 200, response.text
    feedback = response.json()["feedback"]
    assert feedback["dijo"] == "Explicó el inventario del proyecto."
    assert feedback["omisiones"] == ["No mencionó la población"]
    assert feedback["errores_vs_tesis"] == ["Dijo que la muestra era el censo"]
    assert feedback["puntaje"] == 7.5
    assert feedback["muletillas"] == 4
    assert feedback["palabras_por_minuto"] == 2

    slide = response.json()["report"]["slides"][0]
    assert slide["muletillas"] == 4
    assert slide["palabras_por_minuto"] == 2
    assert slide["dijo"] == "Explicó el inventario del proyecto."


def test_global_report_parses_rubric_priorities_and_weak_slides(client, brain):
    prepare_finished_exposition(client)
    brain.evaluator_reply = json.dumps(
        {
            "rubrica": {
                "dominio_del_tema": {"puntaje": 8, "comentario": "Buen dominio"},
                "claridad_y_lenguaje": {"puntaje": 7, "comentario": "Clara"},
                "adecuacion_al_tiempo": {"puntaje": 6, "comentario": "Corta"},
                "coherencia_con_la_tesis": {"puntaje": 9, "comentario": "Coherente"},
                "uso_de_la_presentacion": {"puntaje": 7, "comentario": "Correcto"},
            },
            "prioridades": [
                "Ampliar la metodología",
                "Citar autores",
                "Cerrar con conclusiones",
            ],
            "laminas_debiles": [3, 1, 2],
        }
    )

    response = client.post("/api/feedback/global")

    assert response.status_code == 200, response.text
    report = response.json()["global"]
    assert [item["puntaje"] for item in report["rubrica"]] == [
        8.0,
        7.0,
        6.0,
        9.0,
        7.0,
    ]
    assert report["prioridades"] == [
        "Ampliar la metodología",
        "Citar autores",
        "Cerrar con conclusiones",
    ]
    assert [item["number"] for item in report["laminas_debiles"]] == [3, 1, 2]
    assert report["duration_seconds"] == 20.0
    assert report["within_target"] is False
    assert report["words_per_minute"] > 0
    assert report["muletillas"] == 0
    assert report["parse_error"] is False

    # Persisted report is served by GET /api/feedback and used by the export.
    assert client.get("/api/feedback").json()["report"]["global"] == report
    markdown = client.get("/api/feedback/markdown").text
    assert "## Informe global" in markdown
    assert "Ampliar la metodología" in markdown


def test_global_report_malformed_json_is_defensive(client, brain, stt):
    prepare_finished_exposition(client)
    brain.evaluator_reply = "No pude generar el informe global."

    response = client.post("/api/feedback/global")

    assert response.status_code == 200, response.text
    report = response.json()["global"]
    assert report["parse_error"] is True
    assert report["raw"] == "No pude generar el informe global."
    assert all(item["puntaje"] is None for item in report["rubrica"])
    assert report["prioridades"] == []
    assert report["duration_seconds"] == 20.0
    assert report["muletillas"] == 0
    assert report["words_per_minute"] > 0


def test_global_report_requires_a_finished_exposition(client):
    start_session(client)

    assert client.post("/api/feedback/global").status_code == 409


def test_ui_offers_both_modes_and_real_mode_shortcuts(client):
    html = client.get("/").text
    script = client.get("/static/app.js").text

    assert "Modo práctica" in html
    assert "Defensa real" in html
    assert 'id="expo-mode"' in html
    assert 'id="btn-prev"' in html
    assert "/api/exposition/full/audio" in script
    assert "/api/feedback/global" in script
    for key in ("ArrowRight", "PageDown", "ArrowLeft", "PageUp"):
        assert key in script


def test_ui_renders_the_jury_characters_and_dynamic_turns(client):
    html = client.get("/").text
    script = client.get("/static/app.js").text
    characters = client.get("/static/characters.js").text
    styles = client.get("/static/styles.css").text

    assert "/static/characters.js" in html
    assert 'id="jury-stage"' in html
    assert 'id="tutor-character"' in html
    for role in ("juez_metodologo", "juez_tecnico", "juez_presidente", "tutor"):
        assert role in characters
    assert "createAnalyser" in script
    assert "JuryCharacters" in script
    assert "turn_kind" in script
    assert "current_role" in script
    assert "remaining" in script
    assert "prefers-reduced-motion" in styles
    assert "Sin exposición suficiente" in script or "Sin exposición suficiente" in html


# ---------------------------------------------------------------------------
# Tutor chat
# ---------------------------------------------------------------------------


def test_tutor_chat_reuses_one_brain_session(client, brain):
    start_session(client)

    first = client.post(
        "/api/tutor/message", json={"text": "¿Cómo mejoro la introducción?"}
    )
    assert first.status_code == 200, first.text
    assert first.json()["reply"] == "Tutor responde: ¿Cómo mejoro la introducción?"
    assert first.json()["audio_url"] is None
    assert len(first.json()["messages"]) == 2

    second = client.post("/api/tutor/message", json={"text": "¿Y la metodología?"})
    assert second.status_code == 200
    assert len(second.json()["messages"]) == 4

    tutor_sessions = [role for role in brain.sessions.values() if role == "tutor"]
    assert len(tutor_sessions) == 1

    history = client.get("/api/tutor").json()["messages"]
    assert [message["role"] for message in history] == [
        "user",
        "tutor",
        "user",
        "tutor",
    ]


def test_tutor_audio_message_is_transcribed(client, stt):
    start_session(client)
    stt.text = "pregunta dictada"

    response = client.post(
        "/api/tutor/audio",
        files={"file": ("question.webm", b"fake-question", "audio/webm")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reply"] == "Tutor responde: pregunta dictada"
    assert body["messages"][0]["text"] == "pregunta dictada"


def test_tutor_spoken_reply_returns_audio(client):
    start_session(client)

    response = client.post(
        "/api/tutor/message", json={"text": "Hola", "speak": True}
    )

    assert response.status_code == 200
    audio_url = response.json()["audio_url"]
    assert audio_url and audio_url.startswith("/api/audio/")
    assert client.get(audio_url).status_code == 200


def test_tutor_audio_can_request_a_spoken_reply(client):
    start_session(client)

    response = client.post(
        "/api/tutor/audio",
        files={"file": ("question.webm", b"fake-question", "audio/webm")},
        data={"speak": "true"},
    )

    assert response.status_code == 200, response.text
    audio_url = response.json()["audio_url"]
    assert audio_url and audio_url.startswith("/api/audio/")
    assert client.get(audio_url).status_code == 200


def test_tutor_brain_error_is_visible(client, brain):
    start_session(client)
    brain.fail_roles = {"tutor"}

    response = client.post("/api/tutor/message", json={"text": "Hola"})

    assert response.status_code == 502
    assert "tutor" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def test_extract_json_accepts_plain_fenced_and_embedded_payloads():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Aquí tienes: {"a": {"b": 2}} — listo') == {"a": {"b": 2}}
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_returns_none_for_invalid_text():
    assert extract_json("") is None
    assert extract_json("sin json") is None
    assert extract_json('{"a": 1') is None


# ---------------------------------------------------------------------------
# S13 latency: shared prefix, warm-up, prefetch, fillers and timings
# ---------------------------------------------------------------------------


def test_exposition_start_schedules_one_warmup_per_judge_without_blocking(
    make_client, brain
):
    executor = RecordingExecutor()
    client = make_client(executor=executor, filler_phrases=())
    start_session(client)

    response = client.post("/api/exposition/start", json={"mode": "practice"})

    assert response.status_code == 200, response.text
    assert brain.calls == []
    assert len(executor.tasks) == 1

    client.post("/api/exposition/start", json={"mode": "practice"})
    assert len(executor.tasks) == 1

    executor.run_pending()

    warm_up = [call for call in brain.calls if call[2] == "Responde solo: listo"]
    assert len(warm_up) == 3
    assert {call[1] for call in warm_up} == {
        "juez_metodologo",
        "juez_tecnico",
        "juez_presidente",
    }
    assert executor.tasks == []


def test_jury_consumes_a_prefetched_first_question_once(make_client, brain, tts):
    executor = RecordingExecutor()
    client = make_client(
        executor=executor,
        rng=random.Random(0),
        interjection_probability=0.0,
        filler_phrases=(),
    )
    prepare_finished_exposition(client)

    assert [call for call in brain.calls if call[1].startswith("juez")] == []
    # S23: only the judge who will open the questions is prefetched; the
    # others must hear the first answers before they write their question.
    assert len(executor.tasks) == 1

    executor.run_pending()

    prefetch_calls = [call for call in brain.calls if call[1].startswith("juez")]
    assert len(prefetch_calls) == 1
    assert len(tts.calls) == 1

    response = start_jury(client)

    assert response.status_code == 200, response.text
    jury = response.json()["jury"]
    assert jury["timings"]["prefetched"] is True
    assert jury["current_role"] == prefetch_calls[0][1]
    # No second brain call and no second TTS call for the first question.
    assert len([call for call in brain.calls if call[1].startswith("juez")]) == 1
    assert len(tts.calls) == 1

    follow_up = answer(client, "respuesta")
    assert follow_up.status_code == 200, follow_up.text
    assert follow_up.json()["jury"]["timings"]["prefetched"] is False
    assert follow_up.json()["jury"]["turn_kind"] == "repregunta"


def test_jury_fillers_endpoint_serves_four_clips_per_judge(make_client):
    client = make_client()

    response = client.get("/api/jury/fillers")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {
        "juez_metodologo",
        "juez_tecnico",
        "juez_presidente",
    }
    for role, urls in payload.items():
        assert len(urls) == 4, role
        assert all(url.startswith("/api/audio/fillers/") for url in urls)
        for url in urls:
            audio = client.get(url)
            assert audio.status_code == 200, url
            assert audio.content.startswith(b"RIFF")


def test_jury_payload_and_log_report_per_turn_timings(client, caplog):
    prepare_finished_exposition(client)

    with caplog.at_level(logging.INFO, logger="app.main"):
        response = start_jury(client)

    assert response.status_code == 200, response.text
    timings = response.json()["jury"]["timings"]
    assert timings["kind"] == "pregunta"
    assert timings["role"] in {
        "juez_metodologo",
        "juez_tecnico",
        "juez_presidente",
    }
    assert isinstance(timings["stt_ms"], float | int)
    assert timings["brain_ms"] >= 0
    assert timings["tts_ms"] >= 0
    assert isinstance(timings["prefetched"], bool)
    assert any(
        "brain_ms=" in record.getMessage() and "prefetched=" in record.getMessage()
        for record in caplog.records
    )


def test_jury_audio_answer_reports_stt_timing(client):
    prepare_finished_exposition(client)
    start_jury(client, questions_per_judge=1)

    response = client.post(
        "/api/jury/answer/audio",
        files={"file": ("answer.webm", b"fake-answer-bytes", "audio/webm")},
        data={"turn_id": str(current_jury(client)["current_turn_id"])},
    )

    assert response.status_code == 200, response.text
    timings = response.json()["jury"]["timings"]
    assert timings["stt_ms"] >= 0
    assert timings["stt_ms"] == float(timings["stt_ms"])


# ---------------------------------------------------------------------------
# S15 deliberation, final grade and closing
# ---------------------------------------------------------------------------


def make_defense_session() -> PracticeSession:
    session = PracticeSession(
        id="sess-1",
        created_at=0.0,
        slides=[
            SlideState(
                number=1,
                image="slide-01.png",
                transcript="Expuse el problema y la justificación del proyecto.",
                duration_seconds=125.0,
            ),
            SlideState(
                number=2,
                image="slide-02.png",
                transcript="Expliqué la metodología, la población y el censo.",
                duration_seconds=120.0,
            ),
        ],
    )
    session.jury.judges[0].turns.append(
        JuryTurn(
            question="¿Por qué elegiste ese diseño?",
            answer="Porque el censo era pequeño.",
            kind="pregunta",
            order=1,
        )
    )
    session.jury.judges[1].turns.append(
        JuryTurn(
            question="¿Cómo proteges los datos?",
            answer="Con roles y permisos.",
            kind="pregunta",
            order=2,
        )
    )
    return session


def test_deliberation_prompt_contains_the_whole_defense():
    session = make_defense_session()

    prompt = deliberation_prompt(
        session, session.jury.judges[0], 15, 18, 20
    )

    # Whole exposition, per slide and with its duration.
    assert "Lámina 1" in prompt
    assert "02:05" in prompt
    assert "Expuse el problema y la justificación del proyecto." in prompt
    assert "Lámina 2" in prompt
    assert "02:00" in prompt
    assert "Expliqué la metodología, la población y el censo." in prompt
    # Every question by any judge with its answer, marking the judge's own.
    assert "¿Por qué elegiste ese diseño?" in prompt
    assert "Porque el censo era pequeño." in prompt
    assert "¿Cómo proteges los datos?" in prompt
    assert "Con roles y permisos." in prompt
    assert "tu pregunta" in prompt
    assert "de un colega" in prompt
    # Total time against the 15-18 minute target.
    assert "04:05" in prompt
    assert "15" in prompt and "18" in prompt
    # Strict JSON schema on the 0-20 scale.
    assert '"puntaje_exposicion"' in prompt
    assert '"puntaje_respuestas"' in prompt
    assert '"fortalezas"' in prompt and '"debilidades"' in prompt
    assert '"comentario"' in prompt
    assert "0 a 20" in prompt


def test_jury_deliberation_grades_the_whole_defense(make_client, brain):
    jury_client = make_client(
        rng=random.Random(4), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)

    bodies = answer_until_done(jury_client)
    final = bodies[-1]["result"]["final"]

    assert final["final_grade"] == 17.0
    assert final["average"] == 17.0
    assert final["result"] == "Aprobado"
    assert final["mention"] == "Muy bueno"
    assert final["scale_max"] == 20
    assert final["pass_mark"] == 10
    assert final["weights"] == {"exposicion": 0.5, "respuestas": 0.5}
    assert final["no_exposition"] is False
    assert len(final["verdicts"]) == 3
    assert all(item["nota"] == 17.0 for item in final["verdicts"])
    assert all(item["exposicion"] == 18.0 for item in final["verdicts"])
    assert all(item["respuestas"] == 16.0 for item in final["verdicts"])
    assert len(final["questions"]) == 3
    assert [item["answer"] for item in final["questions"]] == [
        "respuesta 1",
        "respuesta 2",
        "respuesta 3",
    ]

    prompts = [
        call[2]
        for call in brain.calls
        if call[1] == "juez_presidente" and "DELIBERACIÓN FINAL" in call[2]
    ]
    assert len(prompts) == 1
    assert "Lámina 1" in prompts[0] and "Lámina 2" in prompts[0]
    assert "respuesta 1" in prompts[0] and "respuesta 3" in prompts[0]
    assert "00:20" in prompts[0]


def test_jury_enters_deliberation_before_the_final_grade(make_client):
    executor = RecordingExecutor()
    jury_client = make_client(
        rng=random.Random(4),
        interjection_probability=0.0,
        deliberation_executor=executor,
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)
    answer(jury_client, "uno")
    answer(jury_client, "dos")

    third = answer(jury_client, "tres").json()

    assert third["result"]["type"] == "deliberating"
    assert third["result"]["done"] is False
    assert third["result"]["final"] is None
    assert third["jury"]["status"] == "deliberating"
    assert third["jury"]["current_question"] is None

    executor.run_pending()
    jury = current_jury(jury_client)
    assert jury["status"] == "done"
    assert jury["final"]["final_grade"] == 17.0


def test_deliberation_handles_a_malformed_judge_reply(make_client, brain):
    brain.deliberation_replies["juez_tecnico"] = "no pude evaluar nada"
    jury_client = make_client(
        rng=random.Random(4), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)

    final = answer_until_done(jury_client)[-1]["result"]["final"]
    tecnico = next(
        item for item in final["verdicts"] if item["role"] == "juez_tecnico"
    )

    assert tecnico["parse_error"] is True
    assert tecnico["nota"] is None
    assert tecnico["exposicion"] is None
    assert tecnico["respuestas"] is None
    assert len(final["verdicts"]) == 3
    assert final["final_grade"] == 17.0
    assert final["result"] == "Aprobado"


def test_president_reads_the_closing_with_her_voice(make_client, tts):
    jury_client = make_client(
        rng=random.Random(4), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)

    final = answer_until_done(jury_client)[-1]["result"]["final"]

    closing = [
        text
        for text, voice in tts.calls
        if voice == "juez_presidente" and text.startswith("El jurado")
    ]
    assert len(closing) == 1
    assert "17 sobre 20" in closing[0]
    assert "Aprobado" in closing[0]
    assert "Muy bueno" in closing[0]
    assert "claridad al explicar" in closing[0]
    assert not closing[0].rstrip().endswith((" de.", " la.", " el."))
    assert "falta profundizar en la justificación" in closing[0]
    assert len(closing[0].split()) <= 70
    assert final["closing"] == closing[0]
    assert final["closing_audio_url"].startswith("/api/audio/")


def test_verdict_markdown_and_feedback_export(make_client):
    jury_client = make_client(
        rng=random.Random(4), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)
    answer_until_done(jury_client)

    response = jury_client.get("/api/jury/verdict/markdown")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    text = response.text
    assert "# Veredicto del jurado" in text
    assert "17/20" in text
    assert "Aprobado" in text
    assert "Muy bueno" in text
    assert "Evaluación de juez_tecnico" in text
    assert "Fortalezas" in text
    assert "Debilidades" in text
    assert "Preguntas y respuestas" in text
    assert "respuesta 1" in text

    feedback = jury_client.get("/api/feedback/markdown")
    assert feedback.status_code == 200
    assert "Veredicto del jurado" in feedback.text
    assert "17/20" in feedback.text


def test_saved_practice_json_contains_the_final_verdict(make_client, tmp_path):
    jury_client = make_client(
        rng=random.Random(4), interjection_probability=0.0
    )
    prepare_finished_exposition(jury_client)
    start_jury(jury_client, questions_per_judge=1)
    answer_until_done(jury_client)

    saved = jury_client.get("/api/session").json()["saved_path"]
    payload = json.loads(Path(saved).read_text(encoding="utf-8"))

    assert payload["jury"]["final"]["final_grade"] == 17.0
    assert payload["jury"]["final"]["questions"]


def test_ui_binds_answers_to_the_current_turn(client):
    script = client.get("/static/app.js").text

    assert "current_turn_id" in script
    assert "turn_id" in script
    assert "finishing" in script
    assert "ptt.disabled" in script


def test_ui_shows_the_deliberation_and_final_verdict(client):
    html = client.get("/").text
    script = client.get("/static/app.js").text

    assert 'id="jury-deliberating"' in html
    assert "El jurado está deliberando…" in html
    assert "Veredicto del jurado" in html
    assert 'id="jury-verdict-export"' in html
    assert "/api/jury/verdict/markdown" in html
    assert "jury-verdict-export" in script
    assert "deliberating" in script
    assert "verdict-character" in script
    assert "characters.emotionFromScore(item.nota, scale)" in script
    assert "final_grade" in script
    assert "no_exposition" in script


def test_ui_promises_arrow_keys_in_every_mode(client):
    html = client.get("/").text
    script = client.get("/static/app.js").text

    assert "← → para cambiar de lámina" in html
    assert "browseSlide" in script
    assert "previousSlide" in script
    assert "state.pttHeld" in script


# ---------------------------------------------------------------------------
# Direct jury mode (no exposition)
# ---------------------------------------------------------------------------


def test_direct_jury_skips_the_exposition_guard(client, tts):
    start_session(client)
    assert client.post("/api/jury/start", json={"questions_per_judge": 1}).status_code == 409
    assert client.get("/api/session").json()["jury_mode"] == "exposicion"

    response = start_direct_jury(client, questions_per_judge=1)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["jury"]["status"] == "awaiting_answer"
    assert body["jury"]["mode"] == "directo"
    assert body["jury"]["current_role"]
    assert body["jury"]["current_question"]
    assert body["jury"]["timings"]["prefetched"] is True
    assert body["jury"]["greeting"]["text"] == (
        "Buenas tardes, Tu Nombre. Vamos directo a las preguntas."
    )
    assert body["jury"]["greeting"]["audio_url"].startswith("/api/audio/")
    session = client.get("/api/session").json()
    assert session["jury_mode"] == "directo"
    assert session["exposition_status"] == "finished"
    greeting = [
        text
        for text, voice in tts.calls
        if voice == "juez_presidente" and "Vamos directo" in text
    ]
    assert len(greeting) == 1


def test_direct_jury_prompt_skips_the_exposition(client, brain):
    start_session(client)
    start_direct_jury(client, questions_per_judge=1)

    prompt = next(
        call[2]
        for call in brain.calls
        if "no realizó la exposición" in call[2]
        and "DELIBERACIÓN" not in call[2]
    )

    assert "Lámina 1" not in prompt
    assert "presentación" in prompt
    assert "tesis" in prompt
    assert "pregunta" in prompt.lower()


def test_direct_jury_deliberation_grades_only_answers(make_client):
    jury_client = make_client(
        rng=random.Random(4), interjection_probability=0.0
    )
    start_session(jury_client)
    start_direct_jury(jury_client, questions_per_judge=1)

    final = answer_until_done(jury_client)[-1]["result"]["final"]

    assert final["mode"] == "directo"
    assert final["no_exposition"] is True
    assert all(item["exposicion"] is None for item in final["verdicts"])
    assert [item["nota"] for item in final["verdicts"]] == [16.0, 16.0, 16.0]
    assert final["final_grade"] == 16.0
    assert final["result"] == "Aprobado"
    assert final["mention"] == "Muy bueno"
    assert len(final["questions"]) == 3

    markdown = jury_client.get("/api/jury/verdict/markdown").text
    assert "Sin exposición (modo directo)" in markdown
    assert "16/20" in markdown


def test_direct_jury_does_not_block_on_a_per_judge_warmup(client, brain):
    start_session(client)
    start_direct_jury(client, questions_per_judge=1)

    time.sleep(0.2)
    assert not any("Responde solo" in call[2] for call in brain.calls)


def test_ui_offers_a_direct_jury_entry(client):
    html = client.get("/").text
    script = client.get("/static/app.js").text

    assert "Jurado directo" in html
    assert 'id="btn-jury-direct"' in html
    assert 'id="btn-jury-direct-start"' in html
    assert 'id="btn-nav-direct"' in html
    assert "/api/jury/direct" in script
    assert "Sin exposición (modo directo)" in script


def test_direct_mode_ignores_a_model_exposition_score(make_client, brain):
    brain.deliberation_replies = {
        role: json.dumps(
            {
                "puntaje_exposicion": 18,
                "puntaje_respuestas": 16,
                "fortalezas": ["Claridad"],
                "debilidades": ["Tiempo"],
                "comentario": "Buen desempeño.",
            }
        )
        for role in (
            "juez_metodologo",
            "juez_tecnico",
            "juez_presidente",
        )
    }
    jury_client = make_client(
        rng=random.Random(4), interjection_probability=0.0
    )
    start_session(jury_client)
    start_direct_jury(jury_client, questions_per_judge=1)

    final = answer_until_done(jury_client)[-1]["result"]["final"]

    assert final["no_exposition"] is True
    assert all(item["exposicion"] is None for item in final["verdicts"])
    assert [item["nota"] for item in final["verdicts"]] == [16.0, 16.0, 16.0]


# ---------------------------------------------------------------------------
# S19 fluid turns: streaming chunk audio, first-audio latency and deadlines
# ---------------------------------------------------------------------------

STREAMED_QUESTION = (
    "Bien, me queda claro. ¿Por qué elegiste un sistema web en lugar de "
    "seguir con Excel? ¿Cómo lo validaste con la empresa?"
)
BRAIN_DEADLINE_NOTICE = "El juez no respondió a tiempo, seguimos."
JUDGE_ROLE_NAMES = ("juez_metodologo", "juez_tecnico", "juez_presidente")


def test_jury_returns_audio_chunks_before_the_audio_is_ready(
    make_client, brain, tts
):
    brain.judge_reply = STREAMED_QUESTION
    tts.hold = True
    client = make_client(
        prefetch_enabled=False, warmup_enabled=False, filler_phrases=()
    )
    start_session(client)

    response = start_direct_jury(client, questions_per_judge=1)

    assert response.status_code == 200, response.text
    jury = response.json()["jury"]
    assert jury["current_question"] == STREAMED_QUESTION
    chunks = jury["audio_chunks"]
    assert len(chunks) >= 2
    assert all(url.startswith("/api/audio/") for url in chunks)
    # The text came back while every chunk was still being synthesized.
    assert set(tts.pending) == {url.rsplit("/", 1)[-1] for url in chunks}
    assert not [text for text, _role in tts.calls if text in STREAMED_QUESTION]

    threading.Timer(0.05, tts.release).start()
    first = client.get(chunks[0])

    assert first.status_code == 200, first.text
    assert first.content.startswith(b"RIFF")


def test_jury_timings_report_wait_and_first_audio(client):
    prepare_finished_exposition(client)
    start_jury(client, questions_per_judge=1)

    response = answer(client, "mi respuesta")

    assert response.status_code == 200, response.text
    timings = response.json()["jury"]["timings"]
    assert isinstance(timings["wait_ms"], float)
    assert timings["wait_ms"] >= 0
    assert isinstance(timings["first_audio_ms"], float)
    assert timings["first_audio_ms"] >= 0


def test_ui_plays_the_streamed_audio_chunks(client):
    script = client.get("/static/app.js").text

    assert "audio_chunks" in script
    assert "playCharacterAudioList" in script


def test_prefetched_question_is_promoted_without_double_synthesis(
    make_client, brain, tts
):
    brain.judge_reply = STREAMED_QUESTION
    executor = RecordingExecutor()
    client = make_client(
        executor=executor,
        rng=random.Random(0),
        interjection_probability=0.0,
        filler_phrases=(),
        warmup_enabled=False,
    )
    prepare_finished_exposition(client)
    tts.hold = True
    executor.run_pending()

    # The prefetch stored the text and queued the audio at low priority.
    prefetched_requests = list(tts.requests)
    assert len(prefetched_requests) >= 1
    assert {priority for _text, _role, priority in prefetched_requests} == {
        PRIORITY_LOW
    }
    assert not [text for text, _role in tts.calls if text in STREAMED_QUESTION]

    response = start_jury(client)

    assert response.status_code == 200, response.text
    jury = response.json()["jury"]
    assert jury["timings"]["prefetched"] is True
    assert "Bien, me queda claro." in jury["current_question"]
    names = [url.rsplit("/", 1)[-1] for url in jury["audio_chunks"]]
    assert names
    # Every chunk of the consumed prefetch was upgraded to the live priority
    # and none was synthesized twice.
    assert all(tts.priorities[name] == PRIORITY_HIGH for name in names)
    assert not [text for text, _role in tts.calls if text in STREAMED_QUESTION]

    tts.release()

    rendered = Counter(tts.calls)
    assert all(count == 1 for count in rendered.values())
    assert len(rendered) == 2  # two chunks of the one prefetched question


def test_brain_deadline_skips_an_interjection(make_client, brain, caplog):
    client = make_client(
        rng=random.Random(4),
        interjection_probability=1.0,
        prefetch_enabled=False,
        warmup_enabled=False,
        filler_phrases=(),
        deadlines={
            "pregunta": 2.0,
            "repregunta": 2.0,
            "intervencion": 0.05,
            "veredicto": 2.0,
        },
    )
    start_session(client)
    first = start_direct_jury(client, questions_per_judge=1).json()["jury"]
    asker = first["current_role"]
    for role in JUDGE_ROLE_NAMES:
        if role != asker:
            brain.block_asks(role, 1)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = answer(client, "mi respuesta")

    assert time.monotonic() - started < 2.0
    assert response.status_code == 200, response.text
    jury = response.json()["jury"]
    assert jury["status"] == "awaiting_answer"
    assert jury["turn_kind"] == "pregunta"
    assert brain.interrupts
    assert any(
        "action=skip_interjection" in record.getMessage()
        for record in caplog.records
    )


def test_brain_deadline_retries_a_pregunta_in_a_fresh_session(
    make_client, brain, caplog
):
    client = make_client(
        rng=random.Random(4),
        interjection_probability=0.0,
        prefetch_enabled=False,
        warmup_enabled=False,
        filler_phrases=(),
        deadlines={
            "pregunta": 0.05,
            "repregunta": 0.05,
            "intervencion": 0.05,
            "veredicto": 0.05,
        },
    )
    start_session(client)
    brain.block_asks(times=2)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = start_direct_jury(client, questions_per_judge=1)

    assert time.monotonic() - started < 3.0
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["jury"]["status"] == "awaiting_answer"
    assert body["result"]["notice"] == BRAIN_DEADLINE_NOTICE
    # The stalled judge was interrupted and retried in a second session.
    assert len(brain.interrupts) == 2
    stalled_session = brain.interrupts[0]
    stalled_role = brain.sessions[stalled_session]
    assert (
        len(
            [
                role
                for role in brain.sessions.values()
                if role == stalled_role
            ]
        )
        == 2
    )
    assert any(
        "action=retry_fresh_session" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        "action=next_judge" in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Jury questions limited to chosen chapters (S20)
# ---------------------------------------------------------------------------


CHAPTER_CATALOG = [
    {
        "id": "cap1",
        "label": "Capítulo I · El problema",
        "start_line": 178,
        "end_line": 201,
        "sections": [
            {"title": "Planteamiento del Problema", "start_line": 180},
            {"title": "Justificación de la Investigación", "start_line": 199},
        ],
    },
    {
        "id": "cap2",
        "label": "Capítulo II · Marco teórico",
        "start_line": 202,
        "end_line": 337,
        "sections": [
            {"title": "Bases Teóricas", "start_line": 262},
        ],
    },
]


def judge_asks(brain):
    """Every prompt the judges received for a question, never the verdict."""
    return [
        call[2]
        for call in brain.calls
        if call[1].startswith("juez")
        and "DELIBERACIÓN" not in call[2]
    ]


def test_chapters_endpoint_lists_the_catalog(make_client):
    client = make_client(chapters=CHAPTER_CATALOG)

    response = client.get("/api/chapters")

    assert response.status_code == 200, response.text
    chapters = response.json()["chapters"]
    assert [chapter["id"] for chapter in chapters] == ["cap1", "cap2"]
    assert chapters[0]["label"] == "Capítulo I · El problema"


def test_jury_start_accepts_and_stores_chapters(make_client):
    client = make_client(
        chapters=CHAPTER_CATALOG,
        rng=random.Random(0),
        interjection_probability=0.0,
    )
    prepare_finished_exposition(client)

    response = start_jury(client, questions_per_judge=1, chapters=["cap1"])

    assert response.status_code == 200, response.text
    jury = response.json()["jury"]
    assert jury["chapters"] == ["cap1"]
    assert jury["scope_label"] == "Capítulo I · El problema"
    session = client.get("/api/session").json()
    assert session["jury"]["chapters"] == ["cap1"]
    assert session["jury"]["scope_label"] == "Capítulo I · El problema"


def test_direct_jury_accepts_and_stores_chapters(make_client):
    client = make_client(
        chapters=CHAPTER_CATALOG,
        rng=random.Random(0),
        interjection_probability=0.0,
    )
    start_session(client)

    response = start_direct_jury(
        client, questions_per_judge=1, chapters=["cap2"]
    )

    assert response.status_code == 200, response.text
    jury = response.json()["jury"]
    assert jury["chapters"] == ["cap2"]
    assert jury["scope_label"] == "Capítulo II · Marco teórico"


def test_jury_rejects_unknown_chapters(make_client):
    client = make_client(chapters=CHAPTER_CATALOG)
    prepare_finished_exposition(client)

    for path in ("/api/jury/start", "/api/jury/direct"):
        response = client.post(
            path, json={"questions_per_judge": 1, "chapters": ["cap9"]}
        )
        assert response.status_code == 422, path
        detail = response.json()["detail"]
        assert "cap9" in detail
        assert "cap1" in detail and "cap2" in detail

    assert current_jury(client)["status"] == "idle"


def test_scope_block_appears_in_every_judge_prompt_kind():
    scope = chapter_scope_block(["cap1"], CHAPTER_CATALOG)

    assert "Capítulo I · El problema" in scope
    assert "Planteamiento del Problema" in scope
    assert "SOLO" in scope
    prompts = [
        first_question_prompt("transcripción de la exposición", scope=scope),
        next_main_question_prompt(scope=scope),
        follow_up_prompt("respuesta del estudiante", scope=scope),
        interjection_prompt(
            "Dra. Yajaira Contreras", "pregunta", "respuesta", scope=scope
        ),
        direct_first_question_prompt(scope=scope),
    ]
    for prompt in prompts:
        assert "Capítulo I · El problema" in prompt
        assert "Capítulo II · Marco teórico" not in prompt

    assert chapter_scope_block([], CHAPTER_CATALOG) == ""
    assert "Alcance de las preguntas" not in direct_first_question_prompt()


def test_chapter_scope_reaches_every_judge_ask(make_client, brain):
    client = make_client(
        chapters=CHAPTER_CATALOG,
        rng=random.Random(0),
        interjection_probability=0.0,
        prefetch_enabled=False,
        warmup_enabled=False,
    )
    start_session(client)

    response = start_direct_jury(
        client, questions_per_judge=2, chapters=["cap1"]
    )
    assert response.status_code == 200, response.text
    assert response.json()["jury"]["chapters"] == ["cap1"]
    for _ in range(3):
        assert answer(client, "respuesta del estudiante").status_code == 200

    asks = judge_asks(brain)
    assert any("Respuesta del estudiante" in prompt for prompt in asks)
    assert asks
    assert all("Capítulo I · El problema" in prompt for prompt in asks)


def test_no_scope_block_when_all_chapters_are_chosen(make_client, brain):
    client = make_client(
        chapters=CHAPTER_CATALOG,
        rng=random.Random(0),
        interjection_probability=0.0,
        prefetch_enabled=False,
        warmup_enabled=False,
    )
    start_session(client)

    response = start_direct_jury(client, questions_per_judge=1)
    assert response.status_code == 200, response.text
    jury = response.json()["jury"]
    assert jury["chapters"] == []
    assert jury["scope_label"] == "Todos los capítulos"

    asks = judge_asks(brain)
    assert asks
    assert all("Alcance de las preguntas" not in prompt for prompt in asks)


def test_changing_the_chapter_scope_invalidates_pending_prefetches(
    make_client, brain
):
    executor = RecordingExecutor()
    client = make_client(
        chapters=CHAPTER_CATALOG,
        executor=executor,
        rng=random.Random(0),
        interjection_probability=0.0,
        filler_phrases=(),
    )
    prepare_finished_exposition(client)
    assert judge_asks(brain) == []
    executor.run_pending()
    prefetched = judge_asks(brain)
    assert len(prefetched) == 1
    assert all("Alcance de las preguntas" not in prompt for prompt in prefetched)

    response = start_jury(client, questions_per_judge=1, chapters=["cap1"])

    assert response.status_code == 200, response.text
    assert response.json()["jury"]["timings"]["prefetched"] is False
    asks = judge_asks(brain)
    assert len(asks) == 2
    assert "Capítulo I · El problema" in asks[-1]


def test_deliberation_grades_with_the_chapter_scope(make_client, brain):
    client = make_client(
        chapters=CHAPTER_CATALOG,
        rng=random.Random(4),
        interjection_probability=0.0,
        prefetch_enabled=False,
        warmup_enabled=False,
    )
    start_session(client)
    assert (
        start_direct_jury(
            client, questions_per_judge=1, chapters=["cap1"]
        ).status_code
        == 200
    )

    final = answer_until_done(client)[-1]["result"]["final"]

    prompts = [
        call[2] for call in brain.calls if "DELIBERACIÓN FINAL" in call[2]
    ]
    assert len(prompts) == 3
    assert all("Capítulo I · El problema" in prompt for prompt in prompts)
    assert all("solo" in prompt.lower() for prompt in prompts)
    assert final["chapters"] == ["Capítulo I · El problema"]

    markdown = client.get("/api/jury/verdict/markdown").text
    assert "Preguntas sobre: Capítulo I · El problema" in markdown


def test_ui_offers_a_chapter_selector_on_both_jury_entries(client):
    html = client.get("/").text
    script = client.get("/static/app.js").text

    assert 'id="jury-chapter-field"' in html
    assert 'id="jury-chapter-all"' in html
    assert 'id="jury-chapter-options"' in html
    assert 'id="jury-scope"' in html
    assert 'id="jury-final-scope"' in html
    assert "/api/chapters" in script
    assert "selectedJuryChapters" in script
    assert "Preguntas sobre:" in script
    assert "chapters:" in script


# ---------------------------------------------------------------------------
# S21: one slide at a time with the tutor's feedback
# ---------------------------------------------------------------------------

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"


def _decoded_uri(uri):
    header, _, data = uri.partition(",")
    return header, base64.b64decode(data)


def test_slide_feedback_shows_the_slide_image_to_the_evaluator(client, brain):
    start_session(client)
    upload_slide(client, 2)

    response = client.post("/api/feedback/2")

    assert response.status_code == 200, response.text
    role, files = brain.files[-1]
    assert role == "evaluador"
    assert len(files) == 1
    header, data = _decoded_uri(files[0]["uri"])
    assert header == "data:image/png;base64"
    assert data == PNG_BYTES
    prompt = [call[2] for call in brain.calls if call[1] == "evaluador"][-1]
    assert "imagen adjunta" in prompt
    assert "comentario_tutor" in prompt
    assert "máximo 60 palabras" in prompt
    assert "reconocimiento de voz" in prompt


def test_slide_feedback_can_be_spoken_by_the_tutor(client, brain, tts):
    start_session(client)
    upload_slide(client, 1)
    brain.evaluator_reply = json.dumps(
        {
            "slide": 1,
            "dijo": "Presentó el problema",
            "fortalezas": ["Claridad"],
            "correcciones": ["Nombra la empresa"],
            "comentario_tutor": "Muy bien el problema; te faltó nombrar la empresa.",
            "puntaje": 7,
        }
    )

    response = client.post("/api/feedback/1", json={"speak": True})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["feedback"]["comentario_tutor"] == (
        "Muy bien el problema; te faltó nombrar la empresa."
    )
    assert body["spoken_text"] == body["feedback"]["comentario_tutor"]
    assert body["audio_url"].startswith("/api/audio/")
    assert client.get(body["audio_url"]).status_code == 200
    assert tts.calls[-1] == (body["spoken_text"], "tutor")


def test_slide_feedback_speech_falls_back_to_the_lists(client, brain, tts):
    start_session(client)
    upload_slide(client, 1)

    body = client.post("/api/feedback/1", json={"speak": True}).json()

    assert "Claridad" in body["spoken_text"]
    assert "Profundiza en la justificación" in body["spoken_text"]
    assert tts.calls[-1][1] == "tutor"


def test_slide_feedback_without_speak_keeps_the_previous_contract(client, tts):
    start_session(client)
    upload_slide(client, 1)

    body = client.post("/api/feedback/1").json()

    assert body["audio_url"] is None
    assert not any(call[1] == "tutor" for call in tts.calls)
    assert set(("slide", "feedback", "report")) <= set(body)


def test_rerecording_a_slide_clears_its_previous_feedback(client):
    start_session(client)
    upload_slide(client, 1)
    assert client.post("/api/feedback/1").status_code == 200

    upload_slide(client, 1)

    slide = client.get("/api/session").json()["slides"][0]
    assert slide["feedback"] is None


# ---------------------------------------------------------------------------
# S22: the tutor sees slides and images
# ---------------------------------------------------------------------------

def _data_url(data=PNG_BYTES, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def test_tutor_without_attachments_sends_no_files(client, brain):
    start_session(client)

    client.post("/api/tutor/message", json={"text": "Hola"})

    assert brain.files[-1] == ("tutor", [])


def test_tutor_can_see_a_slide(client, brain):
    start_session(client)

    response = client.post(
        "/api/tutor/message", json={"text": "¿Qué mejoro aquí?", "slide": 3}
    )

    assert response.status_code == 200, response.text
    role, files = brain.files[-1]
    assert role == "tutor"
    assert _decoded_uri(files[0]["uri"])[1] == PNG_BYTES
    prompt = brain.calls[-1][2]
    assert "lámina 3" in prompt
    assert "Texto de la lámina 3" in prompt
    assert prompt.rstrip().endswith("¿Qué mejoro aquí?")
    user = response.json()["messages"][-2]
    assert user["slide"] == 3
    assert user["text"] == "¿Qué mejoro aquí?"


def test_tutor_rejects_an_unknown_slide(client):
    start_session(client)

    response = client.post(
        "/api/tutor/message", json={"text": "Hola", "slide": 99}
    )

    assert response.status_code == 404


def test_tutor_receives_and_stores_student_images(client, brain):
    start_session(client)
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg"

    response = client.post(
        "/api/tutor/message",
        json={"text": "", "images": [_data_url(), _data_url(jpeg, "image/jpeg")]},
    )

    assert response.status_code == 200, response.text
    _, files = brain.files[-1]
    assert [_decoded_uri(item["uri"]) for item in files] == [
        ("data:image/png;base64", PNG_BYTES),
        ("data:image/jpeg;base64", jpeg),
    ]
    assert "2 imágenes" in brain.calls[-1][2]
    user = response.json()["messages"][-2]
    assert len(user["images"]) == 2
    stored = client.get(user["images"][1])
    assert stored.status_code == 200
    assert stored.content == jpeg
    assert stored.headers["content-type"] == "image/jpeg"


def test_tutor_rejects_non_image_attachments(client, brain):
    start_session(client)

    response = client.post(
        "/api/tutor/message",
        json={"text": "mira", "images": ["data:text/html;base64,PGI+"]},
    )

    assert response.status_code == 422
    assert not any(call[1] == "tutor" for call in brain.calls)


def test_tutor_still_requires_text_or_an_attachment(client):
    start_session(client)

    response = client.post("/api/tutor/message", json={"text": "  "})

    assert response.status_code == 422


def test_tutor_image_route_rejects_path_traversal(client):
    start_session(client)

    assert client.get("/api/tutor/images/..%2Fsecret.png").status_code == 404


def test_tutor_audio_can_show_a_slide_and_images(client, brain):
    start_session(client)

    response = client.post(
        "/api/tutor/audio",
        files={"file": ("question.webm", b"fake-question", "audio/webm")},
        data={"slide": "2", "images": json.dumps([_data_url()])},
    )

    assert response.status_code == 200, response.text
    _, files = brain.files[-1]
    assert len(files) == 2
    assert "lámina 2" in brain.calls[-1][2]


def test_index_has_the_slide_practice_tab_and_tutor_attachments(client):
    html = client.get("/").text
    for element_id in (
        'id="tab-practice"',
        'id="practice-slide-select"',
        'id="btn-practice-record"',
        'id="practice-feedback"',
        'id="btn-practice-ask-tutor"',
        'id="tutor-attach-input"',
        'id="tutor-slide-select"',
        'id="tutor-attachments"',
    ):
        assert element_id in html


# ---------------------------------------------------------------------------
# S23: the three judges share one room and hear every answer
# ---------------------------------------------------------------------------

def _room_session():
    session = PracticeSession(id="room", created_at=0.0, slides=[])
    technician = next(j for j in session.jury.judges if j.role == "juez_tecnico")
    president = next(j for j in session.jury.judges if j.role == "juez_presidente")
    technician.turns.append(
        JuryTurn(question="¿Por qué un sistema y no Excel?", answer="Por los errores.", order=1)
    )
    president.turns.append(
        JuryTurn(
            question="¿Qué cambia para la gerencia?",
            answer="Ven todo en una pantalla.",
            kind="intervencion",
            order=2,
        )
    )
    return session, technician, president


def test_room_history_lists_every_exchange_in_order():
    session, technician, president = _room_session()

    block = room_history_block(session)

    assert block.index("¿Por qué un sistema y no Excel?") < block.index(
        "¿Qué cambia para la gerencia?"
    )
    assert technician.name in block and president.name in block
    assert "Por los errores." in block and "Ven todo en una pantalla." in block
    assert "misma sala" in block
    assert "No repitas" in block


def test_room_history_is_empty_before_any_answer():
    session = PracticeSession(id="room", created_at=0.0, slides=[])

    assert room_history_block(session) == ""


def test_every_question_prompt_carries_the_room_history():
    history = "HISTORIAL-DE-PRUEBA"
    prompts = [
        first_question_prompt("transcripción", "", history),
        direct_first_question_prompt("", history),
        next_main_question_prompt("", history),
        follow_up_prompt("respuesta", "", history),
        interjection_prompt("Ing. X", "pregunta", "respuesta", "", history),
    ]
    assert all(history in prompt for prompt in prompts)
    # Without history every prompt keeps its previous bytes.
    assert next_main_question_prompt("") == next_main_question_prompt("", "")


def test_next_judge_hears_the_previous_judges_answers(make_client, brain):
    client = make_client(
        prefetch_enabled=False,
        rng=random.Random(0),
        interjection_probability=0.0,
        filler_phrases=(),
    )
    prepare_finished_exposition(client)
    first = start_jury(client).json()["jury"]
    first_role = first["current_role"]
    first_question = first["current_question"]

    answer(client, "respuesta uno sobre excel")
    follow_up = current_jury(client)
    assert follow_up["current_role"] == first_role
    assert "respuesta uno sobre excel" in brain.calls[-1][2]

    body = answer(client, "respuesta dos sobre la base").json()["jury"]

    assert body["current_role"] != first_role
    prompt = [call[2] for call in brain.calls if call[1] == body["current_role"]][-1]
    assert first_question in prompt
    assert "respuesta uno sobre excel" in prompt
    assert "respuesta dos sobre la base" in prompt
    assert "misma sala" in prompt


def test_a_prefetched_question_that_missed_answers_is_not_served(
    make_client, brain
):
    executor = RecordingExecutor()
    client = make_client(
        executor=executor,
        rng=random.Random(0),
        interjection_probability=0.0,
        filler_phrases=(),
    )
    prepare_finished_exposition(client)
    executor.run_pending()
    assert start_jury(client).json()["jury"]["timings"]["prefetched"] is True

    answer(client, "respuesta uno")
    executor.run_pending()
    body = answer(client, "respuesta dos").json()["jury"]
    executor.run_pending()

    assert body["turn_kind"] == "pregunta"
    assert body["timings"]["prefetched"] is False
    prompt = [call[2] for call in brain.calls if call[1] == body["current_role"]][-1]
    assert "respuesta dos" in prompt


def test_progress_endpoint_summarizes_saved_practices(client, tmp_path):
    import json as _json

    practices = tmp_path / "practicas"
    practices.mkdir(exist_ok=True)
    (practices / "old.json").write_text(
        _json.dumps(
            {
                "id": "old",
                "created_at": 10.0,
                "slides": [{"number": 1, "transcript": "o sea", "feedback": {"puntaje": 5}}],
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/api/progress")

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["practices"]] == ["old"]
    assert body["weak_slides"] == [{"number": 1, "average": 5.0, "attempts": 1}]
    assert body["muletillas"] == {"o sea": 1}


def test_rerecording_a_slide_archives_the_previous_attempt(client, brain):
    start_session(client)
    upload_slide(client, 1, duration=60)
    brain.evaluator_reply = json.dumps(
        {
            "puntaje": 5,
            "omisiones": ["No mencionó la población"],
            "correcciones": ["Cita a Arias"],
        }
    )
    assert client.post("/api/feedback/1").status_code == 200

    upload_slide(client, 1, duration=60)

    slide = client.get("/api/session").json()["slides"][0]
    assert slide["feedback"] is None
    assert len(slide["attempts"]) == 1
    attempt = slide["attempts"][0]
    assert attempt["puntaje"] == 5.0
    assert attempt["omisiones"] == ["No mencionó la población"]
    assert attempt["correcciones"] == ["Cita a Arias"]
    assert "muletillas" in attempt and "palabras_por_minuto" in attempt


def test_evaluator_compares_a_retake_with_the_previous_attempt(client, brain):
    start_session(client)
    upload_slide(client, 1)
    brain.evaluator_reply = json.dumps(
        {"puntaje": 5, "omisiones": ["No mencionó la población"], "correcciones": []}
    )
    client.post("/api/feedback/1")
    first_prompt = [call[2] for call in brain.calls if call[1] == "evaluador"][-1]
    assert "intento anterior" not in first_prompt.lower()

    upload_slide(client, 1)
    brain.evaluator_reply = json.dumps(
        {"puntaje": 7, "mejoras_vs_anterior": "Ahora mencionaste la población."}
    )
    response = client.post("/api/feedback/1")

    prompt = [call[2] for call in brain.calls if call[1] == "evaluador"][-1]
    assert "intento anterior" in prompt.lower()
    assert "No mencionó la población" in prompt
    assert '"mejoras_vs_anterior": "..."' in prompt
    assert '"mejoras_vs_anterior"' not in first_prompt
    assert response.json()["feedback"]["mejoras_vs_anterior"] == (
        "Ahora mencionaste la población."
    )
    report_slide = response.json()["report"]["slides"][0]
    assert report_slide["mejoras_vs_anterior"] == "Ahora mencionaste la población."
    assert report_slide["attempts"][0]["puntaje"] == 5.0


def test_slide_status_endpoint_is_a_light_poll(client):
    start_session(client)
    upload_slide(client, 1, duration=12)

    response = client.get("/api/exposition/1/status")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "number",
        "transcription_status",
        "transcription_error",
        "transcript",
        "duration_seconds",
        "has_audio",
    }
    assert body["number"] == 1
    assert body["has_audio"] is True
    assert body["duration_seconds"] == 12.0
    assert client.get("/api/exposition/99/status").status_code == 404


def test_large_json_responses_are_gzip_compressed(client):
    start_session(client)

    response = client.get("/api/session", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"


def test_versioned_slide_images_are_cached_and_stale_versions_revalidate(client):
    start_session(client)
    url = client.get("/api/session").json()["slides"][0]["image_url"]

    fresh = client.get(url)
    stale = client.get(url.split("?")[0] + "?v=1")

    assert fresh.status_code == 200
    assert "immutable" in fresh.headers["cache-control"]
    assert stale.headers["cache-control"] == "no-cache"


def test_two_concurrent_slide_evaluations_both_persist(client):
    from concurrent.futures import ThreadPoolExecutor

    start_session(client)
    for number in (1, 2):
        upload_slide(client, number)

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda n: client.post(f"/api/feedback/{n}"), (1, 2)))

    assert [response.status_code for response in responses] == [200, 200]
    slides = client.get("/api/session").json()["slides"]
    assert all(slide["feedback"] is not None for slide in slides[:2])
