"""Tests for the in-memory practice session state and its JSON persistence."""

import json
import sys
import time
from pathlib import Path

import pytest

# The test command runs from the repository root without a root conftest.py,
# so make the `app` package importable from this file's location.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.session import (  # noqa: E402
    DEFAULT_GRADING,
    EXPOSITION_MODE_PRACTICE,
    GLOBAL_RUBRIC,
    JUDGES,
    JuryState,
    JuryTurn,
    NoSessionError,
    PracticeSession,
    SessionStore,
    SlideMark,
    SlideState,
    TutorMessage,
    apply_full_transcription,
    build_global_report,
    compute_jury_grade,
    count_exposition_words,
    count_muletillas,
    muletillas_breakdown,
    summarize_practices,
    feedback_markdown,
    feedback_report,
    jury_mention,
    jury_verdict_markdown,
    words_per_minute,
)


def make_slide(number: int, text: str = "") -> dict:
    return {
        "number": number,
        "image": f"slide-{number:02d}.png",
        "text": text or f"Texto de la lámina {number}",
    }


def make_store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "practicas")


def test_store_start_builds_session_with_judges(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2)])

    assert store.current is session
    assert session.id
    assert session.save_name
    assert session.exposition_status == "running"
    assert [slide.number for slide in session.slides] == [1, 2]
    assert len(session.jury.judges) == len(JUDGES)
    assert session.jury.questions_per_judge == 2
    assert [judge.role for judge in session.jury.judges] == [
        "juez_metodologo",
        "juez_tecnico",
        "juez_presidente",
    ]
    assert all(judge.name and judge.persona for judge in session.jury.judges)


def test_require_without_session_raises(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(NoSessionError):
        store.require()


def test_session_to_dict_is_json_serializable(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1)])
    session.slides[0].transcript = "transcripción"
    session.slides[0].duration_seconds = 12.5
    session.tutor_messages.append(TutorMessage(role="user", text="hola"))

    payload = session.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    restored = json.loads(encoded)

    assert restored["id"] == session.id
    assert restored["slides"][0]["transcript"] == "transcripción"
    assert restored["tutor_messages"][0]["text"] == "hola"
    assert restored["elapsed_seconds"] >= 0


def test_save_writes_json_and_reuses_same_file(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1)])
    session.slides[0].transcript = "primera"
    first_path = store.save(session)

    session.slides[0].transcript = "segunda"
    second_path = store.save(session)

    assert first_path == second_path
    assert first_path.parent == tmp_path / "practicas"
    assert first_path.suffix == ".json"
    assert session.saved_path == str(first_path)

    saved = json.loads(first_path.read_text(encoding="utf-8"))
    assert saved["slides"][0]["transcript"] == "segunda"
    assert saved["id"] == session.id


def test_exposition_transcript_joins_only_recorded_slides(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2), make_slide(3)])
    session.slides[0].transcript = "Habló de la introducción."
    session.slides[1].transcript = ""
    session.slides[2].transcript = "Habló de los resultados."

    transcript = session.exposition_transcript()

    assert "Lámina 1" in transcript
    assert "Habló de la introducción." in transcript
    assert "Lámina 2" not in transcript
    assert "Lámina 3" in transcript
    assert "Habló de los resultados." in transcript


def test_count_exposition_words_ignores_filler_words(tmp_path):
    assert count_exposition_words("Ok, ok. Eh… este… mm.") == 0
    assert count_exposition_words("La exposición trata ok del inventario") == 5

    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2)])
    session.slides[0].transcript = "Ok, ok"
    session.slides[1].transcript = "La exposición del proyecto"

    assert session.exposition_words() == 4


def test_feedback_report_marks_filler_only_slides_as_insufficient(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2)])
    session.slides[0].transcript = "Ok, ok, eh, mm"
    session.slides[1].transcript = "La exposición del proyecto"

    report = feedback_report(session)

    assert report["slides"][0]["status"] == "insufficient"
    assert "Sin exposición suficiente en esta lámina" in report["slides"][0]["message"]
    assert report["slides"][1]["status"] == "missing"


def test_total_duration_sums_slide_durations(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2)])
    session.slides[0].duration_seconds = 30.0
    session.slides[1].duration_seconds = 12.5

    assert session.total_duration() == 42.5


def test_feedback_report_defaults_and_feedback(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2)])
    session.slides[0].transcript = "texto"
    session.slides[0].duration_seconds = 65.0
    session.slides[0].feedback = {
        "slide": 1,
        "fortalezas": ["Claridad"],
        "correcciones": ["Falta cita"],
        "frase_sugerida": "Como se observa en la figura...",
        "parse_error": False,
    }

    report = feedback_report(session)

    assert report["session_id"] == session.id
    assert report["totals"] == {
        "slides": 2,
        "evaluated": 1,
        "duration_seconds": 65.0,
        "duration_label": "01:05",
    }
    first = report["slides"][0]
    assert first["status"] == "ok"
    assert first["fortalezas"] == ["Claridad"]
    assert first["correcciones"] == ["Falta cita"]
    assert first["frase_sugerida"].startswith("Como se observa")
    assert report["slides"][1]["status"] == "no_transcript"
    assert report["slides"][1]["fortalezas"] == []


def test_feedback_markdown_contains_slides_durations_and_suggestion(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1)])
    session.slides[0].transcript = "texto"
    session.slides[0].duration_seconds = 65.0
    session.slides[0].feedback = {
        "slide": 1,
        "fortalezas": ["Claridad"],
        "correcciones": ["Falta cita"],
        "frase_sugerida": "Como se observa en la figura...",
        "parse_error": False,
    }

    markdown = feedback_markdown(session)

    assert markdown.startswith("# Retroalimentación por lámina")
    assert "## Lámina 1 · 01:05" in markdown
    assert "**Fortalezas**" in markdown
    assert "- Claridad" in markdown
    assert "**Correcciones**" in markdown
    assert "- Falta cita" in markdown
    assert "Como se observa en la figura..." in markdown


def test_feedback_markdown_flags_parse_errors(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1)])
    session.slides[0].transcript = "texto"
    session.slides[0].feedback = {
        "slide": 1,
        "fortalezas": [],
        "correcciones": [],
        "frase_sugerida": "",
        "parse_error": True,
        "raw": "respuesta cruda",
    }

    markdown = feedback_markdown(session)

    assert "No se pudo interpretar la respuesta del evaluador" in markdown
    assert "respuesta cruda" in markdown


def test_elapsed_seconds_tracks_finish_time(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1)])
    session.exposition_started_at = time.time() - 90
    session.exposition_finished_at = session.exposition_started_at + 60

    assert session.elapsed_seconds() == pytest.approx(60.0, abs=0.01)


def test_session_defaults_are_independent(tmp_path):
    store = make_store(tmp_path)
    first = store.start([make_slide(1)])
    second = store.start([make_slide(1)])

    assert first is not second
    assert first.jury.judges is not second.jury.judges
    assert first.slides is not second.slides


def test_jury_state_uses_shared_judge_metadata():
    state = JuryState()

    assert state.status == "idle"
    assert state.current_judge == 0
    assert state.current_role is None
    assert state.turn_kind is None
    assert len(state.judges) == 3
    assert state.judges[0].role == JUDGES[0]["role"]
    assert state.judges[0].name == JUDGES[0]["name"]


def test_jury_turn_keeps_its_kind_and_follow_up_flag():
    turn = JuryTurn(question="q", answer="a", kind="intervencion")

    assert turn.kind == "intervencion"
    assert turn.follow_up is False


def test_practice_session_manual_construction_is_valid():
    session = PracticeSession(
        id="abc12345",
        created_at=time.time(),
        slides=[SlideState(number=1, image="slide-01.png", text="hola")],
    )

    assert session.jury.status == "idle"
    assert session.to_dict()["slides"][0]["number"] == 1
    assert session.exposition_mode == EXPOSITION_MODE_PRACTICE
    assert session.slide_timeline == []


# ---------------------------------------------------------------------------
# Real defense mode: one continuous recording plus a slide timeline
# ---------------------------------------------------------------------------


def test_count_muletillas_counts_filler_words():
    assert count_muletillas("Eh, este... ok, mm. La tesis propone un sistema.") == 4
    assert count_muletillas("La tesis propone un sistema.") == 0
    assert count_muletillas(None) == 0


def test_words_per_minute_uses_duration():
    text = " ".join(["palabra"] * 120)

    assert words_per_minute(text, 60) == 120
    assert words_per_minute(text, 120) == 60
    assert words_per_minute(text, None) is None
    assert words_per_minute(text, 0) is None
    assert words_per_minute("", 60) == 0


def test_apply_full_transcription_assigns_segments_by_midpoint(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2), make_slide(3)])
    timeline = [
        SlideMark(slide=1, start_ms=0),
        SlideMark(slide=2, start_ms=10000),
        SlideMark(slide=3, start_ms=20000),
    ]

    apply_full_transcription(
        session,
        timeline,
        [
            (0.0, 4.0, "introducción"),
            # Starts on slide 1 but its midpoint (10.5 s) belongs to slide 2.
            (9.0, 12.0, "metodología"),
            (21.0, 24.0, "cierre"),
        ],
        total_ms=30000,
    )

    slides = {slide.number: slide for slide in session.slides}
    assert slides[1].transcript == "introducción"
    assert slides[2].transcript == "metodología"
    assert slides[3].transcript == "cierre"
    assert [slides[number].duration_seconds for number in (1, 2, 3)] == [
        10.0,
        10.0,
        10.0,
    ]
    assert all(
        slides[number].transcription_status == "done" for number in (1, 2, 3)
    )


def test_apply_full_transcription_keeps_silent_slides_visible(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2), make_slide(3)])
    timeline = [
        SlideMark(slide=1, start_ms=0),
        SlideMark(slide=2, start_ms=10000),
        SlideMark(slide=3, start_ms=20000),
    ]

    apply_full_transcription(
        session,
        timeline,
        [(0.0, 4.0, "inicio"), (21.0, 24.0, "final")],
        total_ms=30000,
    )

    slides = {slide.number: slide for slide in session.slides}
    assert slides[2].transcript is None
    assert slides[2].transcription_status == "empty"
    assert slides[2].duration_seconds == 10.0


def test_apply_full_transcription_accumulates_a_revisited_slide(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2)])
    timeline = [
        SlideMark(slide=1, start_ms=0),
        SlideMark(slide=2, start_ms=10000),
        SlideMark(slide=1, start_ms=20000),
    ]

    apply_full_transcription(
        session,
        timeline,
        [
            (0.0, 4.0, "primera"),
            (11.0, 14.0, "segunda"),
            (21.0, 24.0, "tercera"),
        ],
        total_ms=30000,
    )

    slides = {slide.number: slide for slide in session.slides}
    assert slides[1].transcript == "primera tercera"
    assert slides[1].duration_seconds == 20.0
    assert slides[2].transcript == "segunda"
    assert slides[2].duration_seconds == 10.0


def test_apply_full_transcription_handles_out_of_order_marks(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2)])
    timeline = [
        SlideMark(slide=2, start_ms=10000),
        SlideMark(slide=1, start_ms=0),
    ]

    apply_full_transcription(
        session,
        timeline,
        [(0.0, 4.0, "inicio"), (11.0, 14.0, "desarrollo")],
        total_ms=20000,
    )

    slides = {slide.number: slide for slide in session.slides}
    assert slides[1].transcript == "inicio"
    assert slides[1].duration_seconds == 10.0
    assert slides[2].transcript == "desarrollo"
    assert slides[2].duration_seconds == 10.0


# ---------------------------------------------------------------------------
# Feedback v2: written fields plus Python-computed metrics and global report
# ---------------------------------------------------------------------------


def test_feedback_report_adds_written_and_computed_fields(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1)])
    session.slides[0].transcript = "Eh, este, el sistema controla el inventario"
    session.slides[0].duration_seconds = 60.0
    session.slides[0].feedback = {
        "slide": 1,
        "dijo": "Resumió el objetivo del sistema.",
        "fortalezas": ["Claridad"],
        "correcciones": ["Falta cita"],
        "omisiones": ["No mencionó la población"],
        "errores_vs_tesis": ["Dijo que la muestra era el censo"],
        "frase_sugerida": "Como se observa en la figura...",
        "puntaje": 7.5,
        "parse_error": False,
    }

    report = feedback_report(session)

    slide = report["slides"][0]
    assert slide["dijo"] == "Resumió el objetivo del sistema."
    assert slide["omisiones"] == ["No mencionó la población"]
    assert slide["errores_vs_tesis"] == ["Dijo que la muestra era el censo"]
    assert slide["puntaje"] == 7.5
    assert slide["muletillas"] == 2
    assert slide["muletillas_detalle"] == {"eh": 1, "este": 1}
    assert slide["palabras_por_minuto"] == 5


def test_build_global_report_uses_rubric_and_python_metrics(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2), make_slide(3)])
    session.slides[0].transcript = "ok ok el sistema de inventario"
    session.slides[0].duration_seconds = 300.0
    session.slides[1].transcript = "este la metodología del proyecto"
    session.slides[1].duration_seconds = 300.0
    session.slides[2].duration_seconds = 300.0
    session.slides[0].feedback = {"puntaje": 6.0}
    session.slides[1].feedback = {"puntaje": 8.0}
    parsed = {
        "rubrica": {
            "dominio_del_tema": {"puntaje": 8, "comentario": "Buen dominio"},
            "claridad_y_lenguaje": {"puntaje": 7, "comentario": "Clara"},
            "adecuacion_al_tiempo": {"puntaje": 6, "comentario": "Ajustada"},
            "coherencia_con_la_tesis": {"puntaje": 9, "comentario": "Coherente"},
            "uso_de_la_presentacion": {"puntaje": 7, "comentario": "Correcto"},
        },
        "prioridades": ["Ampliar la metodología", "Citar autores", "Cerrar mejor", "Extra"],
        "laminas_debiles": [1, 2],
        "parse_error": False,
        "raw": None,
    }

    report = build_global_report(session, parsed, 15, 18)

    assert report["duration_seconds"] == 900.0
    assert report["duration_label"] == "15:00"
    assert report["within_target"] is True
    assert report["muletillas"] == 3
    assert report["muletillas_detalle"] == {"ok": 2, "este": 1}
    assert report["words_per_minute"] == 1
    assert [item["key"] for item in report["rubrica"]] == [
        key for key, _ in GLOBAL_RUBRIC
    ]
    assert report["rubrica"][0]["puntaje"] == 8.0
    assert report["prioridades"] == [
        "Ampliar la metodología",
        "Citar autores",
        "Cerrar mejor",
    ]
    assert [item["number"] for item in report["laminas_debiles"]] == [1, 2]
    assert report["parse_error"] is False


def test_build_global_report_falls_back_to_weakest_scored_slides(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2), make_slide(3)])
    session.slides[0].feedback = {"puntaje": 9.0}
    session.slides[1].feedback = {"puntaje": 4.0}
    session.slides[2].feedback = {"puntaje": 7.0}
    parsed = {
        "rubrica": {},
        "prioridades": [],
        "laminas_debiles": [],
        "parse_error": True,
        "raw": "respuesta cruda",
    }

    report = build_global_report(session, parsed, 15, 18)

    assert [item["number"] for item in report["laminas_debiles"]] == [2, 3, 1]
    assert all(item["puntaje"] is None for item in report["rubrica"])
    assert report["parse_error"] is True
    assert report["raw"] == "respuesta cruda"


def test_feedback_markdown_includes_global_report(tmp_path):
    store = make_store(tmp_path)
    session = store.start([make_slide(1)])
    session.slides[0].transcript = "La exposición del proyecto"
    session.slides[0].duration_seconds = 900.0
    session.slides[0].feedback = {
        "slide": 1,
        "dijo": "Presentó el proyecto.",
        "fortalezas": ["Claridad"],
        "correcciones": ["Falta cita"],
        "omisiones": [],
        "errores_vs_tesis": [],
        "frase_sugerida": "Como se observa...",
        "puntaje": 8.0,
        "parse_error": False,
    }
    session.global_report = build_global_report(
        session,
        {
            "rubrica": {
                "dominio_del_tema": {"puntaje": 8, "comentario": "Buen dominio"},
            },
            "prioridades": ["Citar autores"],
            "laminas_debiles": [1],
            "parse_error": False,
            "raw": None,
        },
        15,
        18,
    )

    markdown = feedback_markdown(session)

    assert markdown.startswith("# Retroalimentación por lámina")
    assert "## Informe global" in markdown
    assert "Tiempo total" in markdown
    assert "Rúbrica" in markdown
    assert "Dominio del tema: 8" in markdown
    assert "Prioridades" in markdown
    assert "Citar autores" in markdown
    assert "Láminas más débiles" in markdown



# ---------------------------------------------------------------------------
# S15 final deliberation and grade (Python-owned math)
# ---------------------------------------------------------------------------


def jury_evaluation(
    role: str,
    name: str,
    exposicion: float | None,
    respuestas: float | None,
) -> dict:
    return {
        "role": role,
        "name": name,
        "exposicion": exposicion,
        "respuestas": respuestas,
        "fortalezas": ["Claridad"],
        "debilidades": ["Tiempo"],
        "comentario": f"Evaluación de {name}",
        "parse_error": False,
    }


def test_jury_grade_uses_the_configured_weights():
    evaluations = [
        jury_evaluation("juez_metodologo", "Ana", 18.0, 16.0),
        jury_evaluation("juez_tecnico", "Beto", 10.0, 10.0),
        jury_evaluation("juez_presidente", "Carla", 20.0, 20.0),
    ]

    grade = compute_jury_grade(evaluations, DEFAULT_GRADING)

    assert [judge["nota"] for judge in grade["judges"]] == [17.0, 10.0, 20.0]
    assert grade["final_grade"] == 15.7
    assert grade["result"] == "Aprobado"
    assert grade["passed"] is True
    assert grade["mention"] == "Muy bueno"
    assert grade["scale_max"] == 20
    assert grade["pass_mark"] == 10
    assert grade["no_exposition"] is False


def test_jury_grade_honors_a_custom_pass_mark():
    evaluations = [
        jury_evaluation("juez_metodologo", "Ana", 10.0, 6.0),
        jury_evaluation("juez_tecnico", "Beto", 10.0, 6.0),
        jury_evaluation("juez_presidente", "Carla", 10.0, 6.0),
    ]

    failed = compute_jury_grade(
        evaluations,
        {"scale_max": 20, "pass_mark": 10, "weights": {"exposicion": 0.5, "respuestas": 0.5}},
    )
    passed = compute_jury_grade(
        evaluations,
        {"scale_max": 20, "pass_mark": 7, "weights": {"exposicion": 0.5, "respuestas": 0.5}},
    )

    assert failed["final_grade"] == 8.0
    assert failed["result"] == "Reprobado"
    assert failed["passed"] is False
    assert failed["mention"] == "Insuficiente"
    assert passed["result"] == "Aprobado"
    assert passed["mention"] == "Suficiente"


@pytest.mark.parametrize(
    ("score", "mention"),
    [
        (20.0, "Excelente"),
        (18.0, "Excelente"),
        (17.9, "Muy bueno"),
        (15.0, "Muy bueno"),
        (14.9, "Bueno"),
        (12.0, "Bueno"),
        (11.9, "Suficiente"),
        (10.0, "Suficiente"),
        (9.9, "Insuficiente"),
    ],
)
def test_jury_mention_thresholds(score, mention):
    assert jury_mention(score, 10) == mention


def test_jury_grade_without_exposition_uses_the_answers_only():
    evaluations = [
        jury_evaluation("juez_metodologo", "Ana", None, 16.0),
        jury_evaluation("juez_tecnico", "Beto", None, 10.0),
        jury_evaluation("juez_presidente", "Carla", None, 14.0),
    ]

    grade = compute_jury_grade(evaluations, DEFAULT_GRADING)

    assert [judge["nota"] for judge in grade["judges"]] == [16.0, 10.0, 14.0]
    assert grade["final_grade"] == 13.3
    assert grade["mention"] == "Bueno"
    assert grade["no_exposition"] is True


def test_jury_grade_skips_a_judge_without_any_score():
    evaluations = [
        jury_evaluation("juez_metodologo", "Ana", 18.0, 16.0),
        jury_evaluation("juez_tecnico", "Beto", None, None),
        jury_evaluation("juez_presidente", "Carla", 18.0, 16.0),
    ]

    grade = compute_jury_grade(evaluations, DEFAULT_GRADING)

    assert grade["judges"][1]["nota"] is None
    assert grade["final_grade"] == 17.0
    assert grade["result"] == "Aprobado"


def make_session_with_final(tmp_path, no_exposition: bool = False):
    store = make_store(tmp_path)
    session = store.start([make_slide(1), make_slide(2)])
    session.jury.final = {
        "final_grade": 17.0,
        "scale_max": 20,
        "pass_mark": 10,
        "result": "Aprobado",
        "mention": "Muy bueno",
        "passed": True,
        "no_exposition": no_exposition,
        "mode": "directo" if no_exposition else "exposicion",
        "closing": "El jurado, luego de deliberar, otorga 17 sobre 20.",
        "verdicts": [
            {
                "role": "juez_presidente",
                "name": "MSc. Carmen Rojas",
                "nota": 17.0,
                "exposicion": None if no_exposition else 18.0,
                "respuestas": 16.0,
                "fortalezas": ["Claridad al explicar"],
                "debilidades": ["Profundizar la justificación"],
                "comentario": "Buen desempeño general.",
                "parse_error": False,
            }
        ],
        "questions": [
            {
                "role": "juez_presidente",
                "judge": "MSc. Carmen Rojas",
                "kind": "pregunta",
                "question": "¿Por qué elegiste ese diseño?",
                "answer": "Porque el censo era pequeño.",
            }
        ],
    }
    return session


def test_jury_verdict_markdown_lists_grade_judges_and_questions(tmp_path):
    session = make_session_with_final(tmp_path)

    markdown = jury_verdict_markdown(session, heading="#")

    assert markdown.startswith("# Veredicto del jurado")
    assert "17/20" in markdown
    assert "Aprobado" in markdown
    assert "Muy bueno" in markdown
    assert "MSc. Carmen Rojas" in markdown
    assert "Exposición: 18/20" in markdown
    assert "Respuestas: 16/20" in markdown
    assert "Fortalezas" in markdown
    assert "Claridad al explicar" in markdown
    assert "Preguntas y respuestas" in markdown
    assert "¿Por qué elegiste ese diseño?" in markdown
    assert "Porque el censo era pequeño." in markdown


def test_jury_verdict_markdown_flags_direct_mode_without_exposition(tmp_path):
    session = make_session_with_final(tmp_path, no_exposition=True)

    markdown = jury_verdict_markdown(session)

    assert "Sin exposición (modo directo)" in markdown
    assert "Exposición: —" in markdown


def test_feedback_markdown_includes_the_jury_verdict(tmp_path):
    session = make_session_with_final(tmp_path)
    session.slides[0].transcript = "La exposición del proyecto"
    session.slides[0].duration_seconds = 900.0

    markdown = feedback_markdown(session)

    assert "# Retroalimentación por lámina" in markdown
    assert "Veredicto del jurado" in markdown
    assert "17/20" in markdown
    assert "¿Por qué elegiste ese diseño?" in markdown


def test_count_muletillas_counts_spanish_filler_phrases():
    text = "O sea, digamos que el sistema, como que, básicamente registra. Ajá, o sea."
    assert count_muletillas(text) == 6
    assert muletillas_breakdown(text) == {
        "o sea": 2,
        "digamos": 1,
        "como que": 1,
        "básicamente": 1,
        "ajá": 1,
    }


def test_muletillas_breakdown_is_ordered_by_frequency_and_empty_without_text():
    assert list(muletillas_breakdown("este, eh, este, este")) == ["este", "eh"]
    assert muletillas_breakdown(None) == {}
    assert muletillas_breakdown("La tesis propone un sistema.") == {}


def test_filler_phrases_do_not_change_the_exposition_word_guard():
    assert count_exposition_words("o sea digamos") == 3


def _write_practice(directory, name, created_at, **fields):
    import json as _json

    payload = {"id": name, "created_at": created_at, "slides": [], **fields}
    (directory / f"{name}.json").write_text(_json.dumps(payload), encoding="utf-8")


def test_summarize_practices_tracks_grades_weak_slides_and_fillers(tmp_path):
    _write_practice(
        tmp_path,
        "a",
        100.0,
        jury_mode="directo",
        jury={"final": {"final_grade": 12.0, "scale_max": 20, "mention": "Bueno"}},
        slides=[
            {"number": 3, "transcript": "o sea el sistema", "feedback": {"puntaje": 4}},
            {"number": 5, "transcript": "este registra", "feedback": {"puntaje": 8}},
        ],
    )
    _write_practice(
        tmp_path,
        "b",
        200.0,
        jury={"final": {"average": 15.0, "scale_max": 20}},
        slides=[{"number": 3, "transcript": "o sea", "feedback": {"puntaje": 6}}],
        global_report={"words_per_minute": 130, "rubrica": [{"puntaje": 8}, {"puntaje": 6}]},
    )
    _write_practice(tmp_path, "empty", 300.0)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    summary = summarize_practices(tmp_path)

    assert [item["id"] for item in summary["practices"]] == ["a", "b"]
    first, second = summary["practices"]
    assert first["jury_grade"] == 12.0
    assert first["scale_max"] == 20
    assert first["slide_average"] == 6.0
    assert first["slides_evaluated"] == 2
    assert first["muletillas"] == 2
    assert second["jury_grade"] == 15.0
    assert second["rubric_average"] == 7.0
    assert second["words_per_minute"] == 130
    assert summary["jury_trend"] == 3.0
    assert summary["weak_slides"][0] == {"number": 3, "average": 5.0, "attempts": 2}
    assert summary["muletillas"] == {"o sea": 2, "este": 1}


def test_summarize_practices_handles_a_missing_directory(tmp_path):
    summary = summarize_practices(tmp_path / "missing")
    assert summary == {
        "practices": [],
        "jury_trend": None,
        "weak_slides": [],
        "muletillas": {},
    }
