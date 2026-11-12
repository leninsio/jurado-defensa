"""FastAPI application for the local jury simulator.

The app serves a single-page UI and a small JSON API. It is deliberately
dependency-injected: ``create_app`` receives the brain, speech-to-text and
text-to-speech objects so the real modules (app/brain.py, app/stt.py,
app/tts.py) can be wired in ``main()`` without importing them here.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import os

import json
import logging
import random
import re
import shutil
import subprocess
import threading
import time
import unicodedata
import uuid
import wave
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.session import (
    DEFAULT_GRADING,
    EXPOSITION_FINISHED,
    EXPOSITION_MODE_PRACTICE,
    EXPOSITION_MODE_REAL,
    EXPOSITION_RUNNING,
    GLOBAL_RUBRIC,
    JUDGE_ROLES,
    JURY_AWAITING_ANSWER,
    JURY_DELIBERATING,
    JURY_DONE,
    JURY_IDLE,
    JURY_MODE_DIRECT,
    JURY_MODE_EXPOSITION,
    TRANSCRIPTION_DONE,
    TRANSCRIPTION_EMPTY,
    TRANSCRIPTION_ERROR,
    TRANSCRIPTION_RUNNING,
    JuryTurn,
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
    format_duration,
    grading_settings,
    jury_verdict_markdown,
    words_per_minute,
)
from app.tts import PRIORITY_HIGH, PRIORITY_LOW, split_speech_chunks

logger = logging.getLogger("app.main")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_QUESTIONS_PER_JUDGE = 2

JURY_TURN_QUESTION = "pregunta"
JURY_TURN_FOLLOW_UP = "repregunta"
JURY_TURN_INTERJECTION = "intervencion"

# T11: a submitted answer is bound to the turn it answers. A stale turn id
# means the same answer already advanced the jury, so it is rejected instead
# of being recorded twice.
DUPLICATE_ANSWER_DETAIL = "Esa pregunta ya fue respondida."

DEFAULT_STUDENT_NAME = "Tu Nombre"
DEFAULT_TUTOR_NAME = "Prof. Andrés Salas"
DEFAULT_OPENING_LINE = (
    "Buenas tardes. Adelante, {name}, ya puedes empezar."
)
DEFAULT_DIRECT_OPENING_LINE = (
    "Buenas tardes, {name}. Vamos directo a las preguntas."
)
DEFAULT_TARGET_MINUTES_MIN = 15
DEFAULT_TARGET_MINUTES_MAX = 18
DEFAULT_INTERJECTION_PROBABILITY = 0.35

# S19: per-turn brain deadlines. A stalled provider turn must never leave the
# student waiting minutes: interrupt the session, retry once, then degrade.
DEFAULT_DEADLINES: dict[str, float] = {
    "pregunta": 25.0,
    "repregunta": 25.0,
    "intervencion": 12.0,
    "veredicto": 60.0,
}
BRAIN_DEADLINE_NOTICE = "El juez no respondió a tiempo, seguimos."
# A chunk audio GET waits this long for a pending chunk file before failing.
AUDIO_WAIT_SECONDS = float(os.environ.get("JURADO_AUDIO_WAIT", "20"))


class BrainDeadline(RuntimeError):
    """A judge turn did not answer before its configured deadline."""

    def __init__(self, role: str, kind: str, seconds: float) -> None:
        super().__init__(f"{role} did not answer a {kind} within {seconds:.1f}s")
        self.role = role
        self.kind = kind
        self.seconds = seconds


class PrefetchTimeout(TimeoutError):
    """The consumed prefetch did not produce its text inside the turn budget."""


@dataclass
class AudioChunk:
    """One sentence chunk of a streamed judge question."""

    text: str
    target: Path


@dataclass
class TurnSpeech:
    """Text plus the queued audio chunks of one spoken judge turn."""

    question: str
    chunks: list[str]
    timing: dict[str, Any]
    first_ready: Future | None = None
    tts_started: float = 0.0


@dataclass
class PrefetchedQuestion:
    """A background question whose text is announced before its audio exists."""

    number: int
    # S23: answered turns the prompt already knew; a later answer makes it stale.
    history_size: int = 0
    started: bool = False
    question: str | None = None
    plan: list[AudioChunk] = None  # type: ignore[assignment]
    brain_ms: float = 0.0
    error: BaseException | None = None
    text_ready: threading.Event = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.plan is None:
            self.plan = []
        if self.text_ready is None:
            self.text_ready = threading.Event()

    def wait_text(self) -> str:
        self.text_ready.wait()
        if self.error is not None:
            raise self.error
        return self.question or ""

    def wait_text_bounded(self, timeout: float) -> str:
        waited = max(0.05, float(timeout))
        if not self.text_ready.wait(waited):
            raise PrefetchTimeout(
                f"prefetched question {self.number} not ready in {waited:.1f}s"
            )
        if self.error is not None:
            raise self.error
        return self.question or ""

# S13 latency: a tiny ask that pays the provider cold-start per judge session.
WARMUP_PROMPT = "Responde solo: listo"
# Neutral reaction clips played while the student submits an answer.
FILLER_PHRASES: tuple[str, ...] = (
    "Mmm, entiendo.",
    "Bien, gracias.",
    "De acuerdo.",
    "Ajá, veamos.",
)


def _audio_slug(role: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", role).strip("_")
    return (slug or "voice")[:40]


def normalize_answer(text: str) -> str:
    """Collapse whitespace and case so an identical retry is detectable."""
    return " ".join((text or "").split()).casefold()


def synthesize_fillers(
    tts,
    out_dir: str | Path,
    roles: tuple[str, ...] | list[str],
    phrases: tuple[str, ...] | list[str] = FILLER_PHRASES,
) -> dict[str, list[str]]:
    """Pre-synthesize one clip per phrase and role; return their audio URLs.

    Existing clips are reused, so restarting the app is cheap. A failing voice
    is logged and skipped: fillers are cosmetic and never break the jury.
    """

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, list[str]] = {}
    for role in roles:
        urls: list[str] = []
        for index, phrase in enumerate(phrases):
            target = directory / f"{_audio_slug(role)}-filler-{index}.wav"
            if not (target.is_file() and target.stat().st_size > 0):
                try:
                    produced = tts.synthesize(
                        phrase, role, directory, priority=PRIORITY_LOW
                    )
                except Exception:  # noqa: BLE001 - cosmetic feature
                    logger.warning(
                        "Filler synthesis failed for role %s", role, exc_info=True
                    )
                    continue
                if not produced:
                    continue
                produced = Path(produced)
                if not produced.is_file() or produced.stat().st_size == 0:
                    continue
                if produced != target:
                    try:
                        produced.replace(target)
                    except OSError:
                        logger.warning(
                            "Could not move filler %s to %s", produced, target
                        )
                        continue
            urls.append(f"/api/audio/fillers/{target.name}")
        mapping[role] = urls
    return mapping


# ---------------------------------------------------------------------------
# Prompts (Spanish: they are sent to the Spanish-speaking judge agents)
# ---------------------------------------------------------------------------


def render_opening_line(template: str, student_name: str) -> str:
    """Render the jury opening line with the student's name."""
    text = (template or DEFAULT_OPENING_LINE).replace(
        "{name}", (student_name or "").strip() or DEFAULT_STUDENT_NAME
    )
    return " ".join(text.split())


def render_direct_greeting(template: str, student_name: str) -> str:
    """Render the direct-jury greeting with the student's name."""
    text = (template or DEFAULT_DIRECT_OPENING_LINE).replace(
        "{name}", (student_name or "").strip() or DEFAULT_STUDENT_NAME
    )
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Chapter scopes (S20)
# ---------------------------------------------------------------------------

DEFAULT_CHAPTERS_FILENAME = "capitulos.json"
ALL_CHAPTERS_LABEL = "Todos los capítulos"


def load_chapter_catalog(path: str | Path | None) -> list[dict[str, Any]]:
    """Read ``capitulos.json`` written by scripts/build_context.py.

    A missing or malformed catalog simply disables the selector; the jury then
    always asks about the whole thesis.
    """
    if not path:
        return []
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read the chapter catalog at %s", path)
        return []
    if not isinstance(raw, list):
        return []
    catalog: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        chapter_id = str(item.get("id") or "").strip()
        label = str(item.get("label") or chapter_id).strip()
        if not chapter_id or not label:
            continue
        sections = [
            {
                "title": str(section.get("title") or "").strip(),
                "start_line": section.get("start_line"),
            }
            for section in item.get("sections") or []
            if isinstance(section, dict) and str(section.get("title") or "").strip()
        ]
        catalog.append(
            {
                "id": chapter_id,
                "label": label,
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
                "sections": sections,
            }
        )
    return catalog


def selected_chapters(
    chapter_ids: list[str] | None, catalog: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The chosen chapters in catalog order (empty means the whole thesis)."""
    if not chapter_ids:
        return []
    wanted = set(chapter_ids)
    return [chapter for chapter in catalog if chapter["id"] in wanted]


def chapter_scope_block(
    chapter_ids: list[str] | None, catalog: list[dict[str, Any]]
) -> str:
    """Scope block prepended to every judge question prompt.

    It names the chosen chapters, their line ranges and their section titles;
    the thesis text is already in the shared context prefix and is not resent.
    """
    chosen = selected_chapters(chapter_ids, catalog)
    if not chosen:
        return ""
    lines = [
        "## Alcance de las preguntas",
        "El estudiante quiere practicar SOLO estos capítulos de la tesis:",
    ]
    for chapter in chosen:
        start, end = chapter.get("start_line"), chapter.get("end_line")
        position = (
            f" (líneas {start}-{end})"
            if start is not None and end is not None
            else ""
        )
        sections = "; ".join(
            section["title"] for section in chapter.get("sections") or []
        )
        lines.append(
            f"- {chapter['label']}{position}"
            + (f": {sections}" if sections else "")
        )
    lines.extend(
        [
            "",
            "Pregunta únicamente sobre el contenido de esos capítulos y aplica "
            "tu especialidad a ese contenido: por ejemplo, si practica el "
            "Capítulo I, el técnico pregunta cómo el problema justifica una "
            "solución de software, no sobre el diseño de la base de datos.",
            "Si la respuesta del estudiante se va a otro capítulo, "
            "reencáusala con amabilidad.",
            "No repitas ni cites el texto de la tesis: ya lo tienes en el "
            "contexto compartido.",
        ]
    )
    return "\n".join(lines)


def _with_scope(prompt: str, scope: str, history: str = "") -> str:
    parts = [part for part in (scope, history, prompt) if part]
    return "\n\n".join(parts)


ROOM_HISTORY_RULES = (
    "Todo el jurado está en la misma sala: tú y tus colegas escucharon cada "
    "pregunta y cada respuesta de esta lista, y son las únicas que realmente se "
    "hicieron. No repitas ni reformules una pregunta que ya hizo cualquier "
    "miembro del jurado, ni preguntes algo que el estudiante ya respondió. Si "
    "vuelves sobre un tema, parte de lo que ya dijo (por ejemplo, «Le dijiste "
    "al ingeniero que…») y pide algo nuevo."
)


def answered_turns(session: PracticeSession) -> int:
    """How many questions the student has answered so far in the defense."""
    return sum(
        1
        for judge in session.jury.judges
        for turn in judge.turns
        if (turn.answer or "").strip()
    )


def room_history_block(session: PracticeSession) -> str:
    """S23: every exchange of the defense so far, shared by the three judges."""
    entries = [
        (judge, turn)
        for _, judge, turn in ordered_turns(session)
        if (turn.answer or "").strip()
    ]
    if not entries:
        return ""
    lines = ["Historial de la sala (preguntas ya hechas y respuestas del estudiante):"]
    for index, (judge, turn) in enumerate(entries, start=1):
        label = JURY_TURN_KIND_LABELS.get(turn.kind, "Pregunta")
        lines.append(f"{index}. {label} de {judge.name}: «{turn.question.strip()}»")
        lines.append(f"   Respuesta del estudiante: «{turn.answer.strip()}»")
    lines.append("")
    lines.append(ROOM_HISTORY_RULES)
    return "\n".join(lines)


DIRECT_QUESTION_PROMPT = (
    "El estudiante no realizó la exposición: iremos directo a las preguntas. "
    "Formula UNA sola pregunta natural sobre su defensa, la tesis o la "
    "presentación en general, desde tu especialidad y en un tono conversacional. "
    "Como no hubo exposición, NO digas \"mencionaste\", \"dijiste\" ni "
    "\"comentaste\": parte de un punto de las láminas o de la tesis (por ejemplo, "
    "\"En tu presentación planteas…\"). "
    "Pregunta por el porqué, el cómo o el para qué; no pidas cifras exactas ni "
    "datos textuales de la tesis. Devuelve únicamente la pregunta, sin "
    "numeración ni comentarios."
)


def direct_first_question_prompt(scope: str = "", history: str = "") -> str:
    """First question when the jury starts without any exposition."""
    return _with_scope(DIRECT_QUESTION_PROMPT, scope, history)


def first_question_prompt(transcript: str, scope: str = "", history: str = "") -> str:
    return _with_scope(
        (
            "Transcripción completa de la exposición del estudiante:\n\n"
            f"{transcript}\n\n"
            "Formula UNA sola pregunta natural, como en una defensa oral real, que parta de "
            "algo concreto que el estudiante dijo en esta exposición y le pida explicar el "
            "porqué o el cómo con sus palabras. No pidas cifras ni datos textuales de la "
            "tesis. Devuelve únicamente la pregunta, sin numeración ni comentarios."
        ),
        scope,
        history,
    )


def follow_up_prompt(answer: str, scope: str = "", history: str = "") -> str:
    return _with_scope(
        (
            "Respuesta del estudiante a tu pregunta anterior:\n\n"
            f"{answer}\n\n"
            "Formula UNA sola pregunta de seguimiento breve y natural sobre lo que acaba de "
            "decir; si respondió bien, profundiza en el porqué, no en datos textuales. "
            "Devuelve únicamente la pregunta."
        ),
        scope,
        history,
    )


def next_main_question_prompt(scope: str = "", history: str = "") -> str:
    return _with_scope(
        (
            "Ya hiciste tu pregunta principal anterior y escuchaste la respuesta del "
            "estudiante. Formula ahora tu SIGUIENTE pregunta principal, natural y "
            "conversacional, a partir de otro punto que el estudiante expuso, distinta "
            "de las que ya hiciste. No pidas cifras ni datos textuales de la tesis. "
            "Devuelve únicamente la pregunta, sin numeración ni comentarios."
        ),
        scope,
        history,
    )


def interjection_prompt(
    colleague: str, question: str, answer: str, scope: str = "", history: str = ""
) -> str:
    return _with_scope(
        (
            f"Estás interviniendo como colega del jurado: {colleague} acaba de hacer "
            "una pregunta y el estudiante ya respondió.\n\n"
            f"Pregunta de {colleague}:\n{question}\n\n"
            f"Respuesta del estudiante:\n{answer}\n\n"
            "Formula UNA sola intervención breve (máximo 40 palabras) sobre esa misma "
            "respuesta: aclara, cuestiona o pide una precisión. Reconoce en una frase "
            "que reaccionas a la pregunta de tu colega. Devuelve únicamente la "
            "intervención, sin numeración ni comentarios."
        ),
        scope,
        history,
    )


JURY_TURN_KIND_LABELS = {
    JURY_TURN_QUESTION: "Pregunta",
    JURY_TURN_FOLLOW_UP: "Repregunta",
    JURY_TURN_INTERJECTION: "Intervención",
}


def ordered_turns(session: PracticeSession) -> list[tuple[int, Any, JuryTurn]]:
    """Every jury turn in the order it happened, with its asking judge."""
    entries = [
        (turn.order, judge, turn)
        for judge in session.jury.judges
        for turn in judge.turns
    ]
    entries.sort(key=lambda item: (item[0], item[1].role))
    return entries


def deliberation_questions(session: PracticeSession) -> list[dict[str, Any]]:
    """Serialize every question and answer of the defense for the verdict UI."""
    return [
        {
            "role": judge.role,
            "judge": judge.name,
            "kind": turn.kind,
            "question": turn.question,
            "answer": turn.answer,
        }
        for _, judge, turn in ordered_turns(session)
    ]


def deliberation_prompt(
    session: PracticeSession,
    judge,
    target_minutes_min: int = DEFAULT_TARGET_MINUTES_MIN,
    target_minutes_max: int = DEFAULT_TARGET_MINUTES_MAX,
    scale_max: float = 20,
    scope_label: str = "",
) -> str:
    """Prompt that asks one judge to grade the WHOLE defense (S15).

    It carries the full exposition per slide with its duration, every question
    asked by any judge with the student's answer (marking the judge's own), and
    the total time against the target. The final grade is never computed here.
    """
    duration = session.total_duration()
    min_seconds = target_minutes_min * 60
    max_seconds = target_minutes_max * 60
    within_target = bool(
        duration > 0 and min_seconds <= duration <= max_seconds
    )
    direct = session.jury_mode == JURY_MODE_DIRECT
    scale_label = f"{scale_max:g}"

    lines = [
        "Es el momento de la DELIBERACIÓN FINAL. Evalúa TODA la defensa del "
        "estudiante, no una sola respuesta.",
        "",
        f"Duración total: {format_duration(duration)} sobre un objetivo de "
        f"{target_minutes_min} a {target_minutes_max} minutos "
        f"({'dentro' if within_target else 'fuera'} del objetivo).",
        "",
    ]
    if direct:
        lines.extend(
            [
                "El estudiante no realizó la exposición (modo directo): "
                "califica solo las respuestas.",
                "",
            ]
        )
    if scope_label:
        lines.extend(
            [
                "Alcance de la práctica: el estudiante pidió preguntas solo sobre "
                f"{scope_label}, así que evaluaste solo ese contenido.",
                "No lo penalices por no cubrir los demás capítulos de la tesis: "
                "califica lo que mostró dentro de ese alcance.",
                "",
            ]
        )
    lines.append("Exposición completa, lámina por lámina:")
    lines.append("")
    exposed = False
    for slide in session.slides:
        transcript = (slide.transcript or "").strip()
        if not transcript:
            continue
        exposed = True
        lines.append(
            f"Lámina {slide.number} ({format_duration(slide.duration_seconds)}):"
        )
        lines.append(transcript)
        lines.append("")
    if not exposed:
        lines.append("Sin exposición registrada.")
        lines.append("")

    lines.append(
        "Preguntas del jurado y respuestas del estudiante (marca cuáles "
        "fueron tuyas):"
    )
    entries = ordered_turns(session)
    if not entries:
        lines.append("- Sin preguntas registradas.")
    for index, (_, asker, turn) in enumerate(entries, start=1):
        marker = (
            "tu pregunta" if asker.role == judge.role else "de un colega"
        )
        kind = JURY_TURN_KIND_LABELS.get(turn.kind, "Pregunta")
        lines.append(
            f"{index}. [{marker} · {asker.name} · {asker.role}] {kind}"
        )
        lines.append(f"   P: {turn.question}")
        lines.append(f"   R: {turn.answer}")
    lines.append("")

    exposition_score = "null" if direct else "0"
    lines.extend(
        [
            "Devuelve SOLO un objeto JSON con esta forma:",
            '{"puntaje_exposicion": '
            f'{exposition_score}, "puntaje_respuestas": 0, '
            '"fortalezas": ["...", "..."], '
            '"debilidades": ["...", "..."], '
            '"comentario": "2 o 3 frases dirigidas al estudiante"}',
            f"Cada puntaje va de 0 a {scale_label}.",
            "Incluye 2 o 3 fortalezas y 2 o 3 debilidades concretas, y un "
            "comentario de 2 o 3 frases dirigido al estudiante.",
            "Evalúa desde tu especialidad y el rol que tienes en el jurado.",
            "No inventes datos, cifras ni citas de la tesis.",
        ]
    )
    if direct:
        lines.append(
            'En este modo no hubo exposición: devuelve "puntaje_exposicion": '
            "null."
        )
    return "\n".join(lines)


def _shorten(text: Any, max_words: int) -> str:
    words = str(text or "").split()
    return " ".join(words[:max_words]).strip(" .,;:")


def _first_sentence(text: Any) -> str:
    """Return the first complete sentence, never cut mid-phrase."""
    value = " ".join(str(text or "").split())
    match = re.search(r"^(.+?[.!?])(\s|$)", value)
    sentence = match.group(1) if match else value
    sentence = sentence.rstrip(" .;:")
    return sentence[:1].lower() + sentence[1:] if sentence else sentence


def build_closing_line(final: dict[str, Any]) -> str:
    """Build the president's spoken closing from the three evaluations."""
    grade = final.get("final_grade")
    scale = final.get("scale_max") or 20
    if grade is None:
        return (
            "El jurado, luego de deliberar, no pudo emitir una calificación. "
            "Repite la defensa o revisa la conexión."
        )
    result = final.get("result") or ""
    mention = final.get("mention") or ""
    head = (
        "El jurado, luego de deliberar, otorga la calificación de "
        f"{grade:g} sobre {scale:g}: {result}"
    )
    if mention:
        head += f", {mention}"
    parts = [head + "."]
    verdicts = final.get("verdicts") or []
    strength = next(
        (
            entry
            for item in verdicts
            for entry in (item.get("fortalezas") or [])
            if str(entry).strip()
        ),
        "",
    )
    weakness = next(
        (
            entry
            for item in verdicts
            for entry in (item.get("debilidades") or [])
            if str(entry).strip()
        ),
        "",
    )
    if strength:
        parts.append(f"Como fortaleza, el jurado destaca: {_first_sentence(strength)}.")
    if weakness:
        parts.append(f"Como aspecto por mejorar: {_first_sentence(weakness)}.")
    return " ".join(parts)

def previous_attempt_block(previous: dict[str, Any] | None) -> str:
    """Tell the evaluator what the last take missed so it can compare."""
    if not previous:
        return ""
    lines = ["Intento anterior de esta lámina:"]
    if previous.get("puntaje") is not None:
        lines.append(f"- Puntaje: {previous['puntaje']}/10")
    for label, key in (("Omisiones", "omisiones"), ("Correcciones", "correcciones")):
        items = previous.get(key) or []
        if items:
            lines.append(f"- {label}: " + "; ".join(str(item) for item in items))
    lines.append(
        "Compara esta toma con el intento anterior y en mejoras_vs_anterior "
        "di en una oración qué corrigió y qué sigue faltando.\n"
    )
    return "\n".join(lines) + "\n"


def evaluator_prompt(
    number: int,
    slide_text: str,
    transcript: str,
    has_image: bool = False,
    previous: dict[str, Any] | None = None,
) -> str:
    previous_block = previous_attempt_block(previous)
    image_line = (
        "La imagen adjunta es esta lámina tal como la ve el jurado; úsala "
        "para juzgar si el estudiante explicó lo que muestra.\n\n"
        if has_image
        else ""
    )
    return (
        f"Lámina {number}:\n{slide_text}\n\n"
        f"{image_line}"
        "Transcripción de lo que expuso el estudiante en esta lámina:\n"
        f"{transcript}\n\n"
        f"{previous_block}"
        "Devuelve SOLO un objeto JSON con esta forma:\n"
        f'{{"slide": {number}, "dijo": "resumen de una línea de lo que dijo", '
        '"fortalezas": ["..."], "correcciones": ["..."], '
        '"omisiones": ["puntos clave de la lámina o de la tesis que no mencionó"], '
        '"errores_vs_tesis": ["afirmaciones que contradicen la tesis, '
        'citando la transcripción entre comillas"], '
        '"frase_sugerida": "...", '
        '"comentario_tutor": "2 o 3 oraciones habladas, de tú, como su tutor: '
        'primero lo que dijo bien y luego lo que dijo mal o le faltó", '
        + ('"mejoras_vs_anterior": "...", ' if previous_block else "")
        + '"puntaje": 0}\n'
        "El puntaje va de 0 a 10. No calcules muletillas ni palabras por "
        "minuto: la aplicación las calcula.\n"
        "El comentario_tutor tiene como máximo 60 palabras, porque se lee en voz "
        "alta.\n"
        "La transcripción es automática: un nombre o término mal escrito que "
        "suena parecido al correcto (por ejemplo «QD Richardson» por "
        "Kuder-Richardson) es un error del reconocimiento de voz, no del "
        "estudiante; no lo cuentes como error ni lo corrijas."
    )


# S22: images the tutor accepts from the student.
IMAGE_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
SUFFIX_IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
MAX_TUTOR_IMAGES = 4
MAX_TUTOR_IMAGE_BYTES = 8 * 1024 * 1024


def image_attachment(path: Path) -> dict[str, str]:
    """Build an OpenCode prompt attachment that inlines ``path`` as a data URI."""
    mime = SUFFIX_IMAGE_MIMES.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"uri": f"data:{mime};base64,{encoded}", "name": path.name}


def decode_image_data_url(value: str) -> tuple[str, bytes]:
    """Return ``(mime, bytes)`` of a base64 image data URL or raise ValueError."""
    header, separator, payload = (value or "").partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(
        ";base64"
    ):
        raise ValueError("La imagen debe llegar como data URL en base64.")
    mime = header[len("data:") : -len(";base64")].lower()
    if mime not in IMAGE_MIME_SUFFIXES:
        raise ValueError("Solo se aceptan imágenes PNG, JPG, WEBP o GIF.")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("La imagen está dañada.") from exc
    if not data:
        raise ValueError("La imagen llegó vacía.")
    if len(data) > MAX_TUTOR_IMAGE_BYTES:
        raise ValueError("Cada imagen debe pesar menos de 8 MB.")
    return mime, data


def tutor_feedback_speech(feedback: dict[str, Any]) -> str:
    """Short spoken summary of one slide's feedback, in the tutor's voice."""
    comment = str(feedback.get("comentario_tutor") or "").strip()
    if comment:
        return comment
    parts: list[str] = []
    strengths = feedback.get("fortalezas") or []
    corrections = feedback.get("correcciones") or []
    errors = feedback.get("errores_vs_tesis") or []
    if strengths:
        parts.append(f"Lo que dijiste bien: {strengths[0].rstrip('.')}.")
    if corrections:
        parts.append(f"Lo que debes corregir: {corrections[0].rstrip('.')}.")
    if errors:
        parts.append(f"Ojo con esto: {errors[0].rstrip('.')}.")
    if not parts:
        return "No alcancé a evaluar esta lámina; inténtala de nuevo."
    return " ".join(parts)


def tutor_prompt(text: str, slide: SlideState | None, image_count: int) -> str:
    """Prefix the student's question with what they are showing the tutor."""
    parts: list[str] = []
    if slide is not None:
        parts.append(
            f"[Te muestro la lámina {slide.number} de mi presentación "
            f"(imagen adjunta). Su texto es:\n{slide.text}]"
        )
    if image_count:
        noun = "imagen" if image_count == 1 else "imágenes"
        parts.append(f"[Te adjunto {image_count} {noun}.]")
    parts.append(text or "¿Qué opinas de lo que te muestro?")
    return "\n\n".join(parts)


def global_feedback_prompt(
    session: PracticeSession,
    target_minutes_min: int,
    target_minutes_max: int,
) -> str:
    report = feedback_report(session)
    duration = session.total_duration()
    total_words = session.exposition_words()
    wpm = (
        int(round(total_words / (duration / 60.0))) if duration > 0 else None
    )
    muletillas = sum(
        count_muletillas(slide.transcript) for slide in session.slides
    )
    lines = [
        "Datos objetivos de la práctica (calculados por la aplicación):",
        f"- Duración total: {report['totals']['duration_label']} "
        f"(objetivo {target_minutes_min}–{target_minutes_max} minutos)",
        f"- Palabras por minuto: {wpm if wpm is not None else '—'}",
        f"- Muletillas: {muletillas}",
        "",
        "Resumen por lámina:",
    ]
    for slide in report["slides"]:
        transcript = (slide["transcript"] or "").strip()
        if not transcript:
            continue
        lines.append(
            f"Lámina {slide['number']} ({slide['duration_label']}):"
        )
        if slide["dijo"]:
            lines.append(f"- Dijo: {slide['dijo']}")
        else:
            lines.append(f"- Dijo: {transcript[:400]}")
        if slide["puntaje"] is not None:
            lines.append(f"- Puntaje de la lámina: {slide['puntaje']}")
        if slide["fortalezas"]:
            lines.append("- Fortalezas: " + "; ".join(slide["fortalezas"]))
        if slide["correcciones"]:
            lines.append(
                "- Correcciones: " + "; ".join(slide["correcciones"])
            )
        if slide["omisiones"]:
            lines.append("- Omisiones: " + "; ".join(slide["omisiones"]))
        if slide["errores_vs_tesis"]:
            lines.append(
                "- Errores vs. la tesis: "
                + "; ".join(slide["errores_vs_tesis"])
            )
        lines.append("")
    lines.extend(
        [
            "Devuelve SOLO un objeto JSON con esta forma:",
            '{"rubrica": {'
            '"dominio_del_tema": {"puntaje": 0, "comentario": "..."}, '
            '"claridad_y_lenguaje": {"puntaje": 0, "comentario": "..."}, '
            '"adecuacion_al_tiempo": {"puntaje": 0, "comentario": "..."}, '
            '"coherencia_con_la_tesis": {"puntaje": 0, "comentario": "..."}, '
            '"uso_de_la_presentacion": {"puntaje": 0, "comentario": "..."}}, '
            '"prioridades": ["...", "...", "..."], '
            '"laminas_debiles": [0, 0, 0]}',
            "Cada puntaje va de 0 a 10. Evalúa con los criterios de la "
            "Unidad III del CUAM.",
            "No calcules tiempos, palabras por minuto ni muletillas: ya "
            "están calculados.",
            "No inventes datos ni citas de la tesis.",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Any | None:
    """Extract the first JSON value from a model reply.

    Handles plain JSON, JSON inside a Markdown fence, and JSON embedded in
    prose. Returns ``None`` when no valid JSON value can be found.
    """
    if not text or not text.strip():
        return None
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except (json.JSONDecodeError, TypeError):
            continue

    return _scan_balanced_json(candidate)


def _scan_balanced_json(text: str) -> Any | None:
    """Find the first balanced object/array in free text and decode it."""
    start = next((index for index, char in enumerate(text) if char in "{["), None)
    if start is None:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            expected = "{" if char == "}" else "["
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def load_slides(slides_dir: str | Path) -> list[dict[str, Any]]:
    """List ``slide-NN.png`` files sorted by name and pair them with ``.txt``."""
    directory = Path(slides_dir)
    if not directory.is_dir():
        logger.warning("Slides directory %s does not exist", directory)
        return []
    slides: list[dict[str, Any]] = []
    for position, image in enumerate(sorted(directory.glob("slide-*.png")), start=1):
        match = re.search(r"(\d+)$", image.stem)
        number = int(match.group(1)) if match else position
        text_path = image.with_suffix(".txt")
        text = ""
        if text_path.is_file():
            text = text_path.read_text(encoding="utf-8").strip()
        slides.append({"number": number, "image": image.name, "text": text})
    return slides


def parse_evaluator_reply(raw: str, slide_number: int) -> dict[str, Any]:
    """Parse the evaluator JSON defensively, keeping the raw text on failure."""
    data = extract_json(raw)
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), None)
    if not isinstance(data, dict):
        return {
            "slide": slide_number,
            "dijo": "",
            "fortalezas": [],
            "correcciones": [],
            "omisiones": [],
            "errores_vs_tesis": [],
            "frase_sugerida": "",
            "comentario_tutor": "",
            "puntaje": None,
            "parse_error": True,
            "raw": (raw or "").strip(),
        }

    def as_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    suggestion = data.get("frase_sugerida") or data.get("fraseSugerida") or ""
    return {
        "slide": slide_number,
        "dijo": str(data.get("dijo") or data.get("resumen") or "").strip(),
        "fortalezas": as_list(data.get("fortalezas")),
        "correcciones": as_list(data.get("correcciones")),
        "omisiones": as_list(data.get("omisiones")),
        "errores_vs_tesis": as_list(
            data.get("errores_vs_tesis") or data.get("erroresVsTesis")
        ),
        "frase_sugerida": str(suggestion).strip(),
        "comentario_tutor": str(
            data.get("comentario_tutor") or data.get("comentarioTutor") or ""
        ).strip(),
        "mejoras_vs_anterior": str(data.get("mejoras_vs_anterior") or "").strip(),
        "puntaje": _score_value(data.get("puntaje", data.get("score"))),
        "parse_error": False,
        "raw": None,
    }


def _score_value(value: Any) -> float | None:
    """Coerce a model-provided score to a 0-10 float, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return round(min(10.0, max(0.0, score)), 1)


def _slug(value: Any) -> str:
    text = (
        unicodedata.normalize("NFKD", str(value or ""))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_CRITERION_KEYS = {_slug(key): key for key, _ in GLOBAL_RUBRIC}


def parse_global_reply(raw: str) -> dict[str, Any]:
    """Parse the evaluator's global report, tolerating fenced or embedded JSON."""
    data = extract_json(raw)
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), None)
    rubrica: dict[str, dict[str, Any]] = {}
    prioridades: list[str] = []
    laminas: list[int] = []
    if isinstance(data, dict):
        rubric = data.get("rubrica") or data.get("rubric") or {}
        if isinstance(rubric, dict):
            for name, value in rubric.items():
                key = _CRITERION_KEYS.get(_slug(name))
                if key is None:
                    continue
                if isinstance(value, dict):
                    rubrica[key] = {
                        "puntaje": value.get("puntaje", value.get("score")),
                        "comentario": str(
                            value.get("comentario")
                            or value.get("comment")
                            or ""
                        ).strip(),
                    }
                else:
                    rubrica[key] = {"puntaje": value, "comentario": ""}
        elif isinstance(rubric, list):
            for item in rubric:
                if not isinstance(item, dict):
                    continue
                key = _CRITERION_KEYS.get(
                    _slug(
                        item.get("criterio")
                        or item.get("key")
                        or item.get("nombre")
                    )
                )
                if key is None:
                    continue
                rubrica[key] = {
                    "puntaje": item.get("puntaje", item.get("score")),
                    "comentario": str(
                        item.get("comentario") or item.get("comment") or ""
                    ).strip(),
                }
        prioridades = _string_list(
            data.get("prioridades") or data.get("priorities")
        )
        laminas = _int_list(
            data.get("laminas_debiles")
            or data.get("diapositivas_debiles")
            or data.get("weak_slides")
        )
    parsed = isinstance(data, dict)
    return {
        "rubrica": rubrica,
        "prioridades": prioridades,
        "laminas_debiles": laminas,
        "parse_error": not parsed,
        "raw": None if parsed else (raw or "").strip(),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    numbers: list[int] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("slide", item.get("number"))
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            continue
    return numbers


def parse_timeline(raw: str, valid_slides: set[int]) -> list[SlideMark]:
    """Parse the real-mode slide timeline defensively.

    Expects a JSON list of ``{"slide": n, "start_ms": ms}``. Unknown slides,
    negative timestamps, and malformed entries are dropped. Raises
    ``ValueError`` when no valid mark remains.
    """
    try:
        data = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("La línea de tiempo no es JSON válido.") from exc
    if not isinstance(data, list):
        raise ValueError("La línea de tiempo debe ser una lista de láminas.")
    marks: list[SlideMark] = []
    seen: set[tuple[int, int]] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("slide"))
            start_ms = int(round(float(item.get("start_ms"))))
        except (TypeError, ValueError):
            continue
        if number not in valid_slides or start_ms < 0:
            continue
        key = (number, start_ms)
        if key in seen:
            continue
        seen.add(key)
        marks.append(SlideMark(slide=number, start_ms=start_ms))
    marks.sort(key=lambda mark: mark.start_ms)
    if not marks:
        raise ValueError("La línea de tiempo no tiene láminas válidas.")
    return marks


def wav_duration_seconds(path: str | Path) -> float | None:
    """Read a wav file's duration, or ``None`` when it is not a readable wav."""
    try:
        with wave.open(str(path)) as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (wave.Error, EOFError, OSError):
        return None
    if not rate:
        return None
    return frames / float(rate)



def _deliberation_score(value: Any, scale_max: float) -> float | None:
    """Coerce a model-provided 0-``scale_max`` score, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return round(min(float(scale_max), max(0.0, score)), 1)


def parse_deliberation_reply(
    raw: str, scale_max: float = 20
) -> dict[str, Any]:
    """Parse one judge's deliberation JSON defensively.

    A malformed reply never invents a score: both components stay ``None`` and
    the final grade skips that judge. The raw text is kept as the comment so
    the student can still see what the model answered.
    """
    data = extract_json(raw or "")
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), None)
    if not isinstance(data, dict):
        return {
            "exposicion": None,
            "respuestas": None,
            "fortalezas": [],
            "debilidades": [],
            "comentario": (raw or "").strip(),
            "parse_error": True,
        }

    def as_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    return {
        "exposicion": _deliberation_score(
            data.get("puntaje_exposicion", data.get("exposicion")), scale_max
        ),
        "respuestas": _deliberation_score(
            data.get("puntaje_respuestas", data.get("respuestas")), scale_max
        ),
        "fortalezas": as_list(data.get("fortalezas"))[:3],
        "debilidades": as_list(data.get("debilidades"))[:3],
        "comentario": str(
            data.get("comentario") or data.get("comment") or ""
        ).strip(),
        "parse_error": False,
    }


def _clean_reply(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().strip('"')


def _convert_to_wav(source: Path) -> Path:
    """Best-effort ffmpeg conversion to 16 kHz mono wav for the STT engine."""
    if source.suffix.lower() == ".wav":
        return source
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; passing %s to STT unchanged", source)
        return source
    target = source.with_suffix(".wav")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(target),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return target
    except Exception:  # pragma: no cover - depends on the local ffmpeg build
        logger.exception("ffmpeg conversion failed for %s", source)
        return source


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class NewSessionRequest(BaseModel):
    questions_per_judge: int = 2


class JuryStartRequest(BaseModel):
    questions_per_judge: int | None = None
    # S20: chapter ids to limit the questions to; empty/missing means all.
    chapters: list[str] | None = None


class ExpositionStartRequest(BaseModel):
    mode: str | None = None


class TextAnswerRequest(BaseModel):
    text: str
    turn_id: int | None = None


class TutorMessageRequest(BaseModel):
    text: str = ""
    speak: bool = False
    slide: int | None = None
    images: list[str] = []


class SlideFeedbackRequest(BaseModel):
    speak: bool = False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


class SelectiveGZip:
    """GZip for API JSON and static files; audio and slide images pass through."""

    SKIP_PREFIXES = ("/api/audio/", "/api/slides/")

    def __init__(self, app):
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=1024)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not scope["path"].startswith(self.SKIP_PREFIXES):
            await self.gzip(scope, receive, send)
        else:
            await self.app(scope, receive, send)


def create_app(
    brain,
    stt,
    tts,
    slides_dir: str | Path,
    audio_dir: str | Path,
    session_dir: str | Path | None = None,
    min_exposition_words: int = 80,
    student_name: str = DEFAULT_STUDENT_NAME,
    opening_line: str = DEFAULT_OPENING_LINE,
    target_minutes_min: int = DEFAULT_TARGET_MINUTES_MIN,
    target_minutes_max: int = DEFAULT_TARGET_MINUTES_MAX,
    interjection_probability: float = DEFAULT_INTERJECTION_PROBABILITY,
    rng: random.Random | None = None,
    tutor_name: str = DEFAULT_TUTOR_NAME,
    executor: Executor | None = None,
    warmup_enabled: bool = True,
    prefetch_enabled: bool = True,
    filler_phrases: tuple[str, ...] = FILLER_PHRASES,
    prepare_fillers_on_startup: bool = False,
    grading: dict[str, Any] | None = None,
    direct_opening_line: str = DEFAULT_DIRECT_OPENING_LINE,
    deliberation_executor: Executor | None = None,
    deadlines: dict[str, float] | None = None,
    chapters: list[dict[str, Any]] | None = None,
) -> FastAPI:
    """Build the FastAPI app around injected brain/STT/TTS implementations."""
    slides_path = Path(slides_dir)
    audio_path = Path(audio_dir)
    audio_path.mkdir(parents=True, exist_ok=True)
    tutor_images_path = audio_path / "tutor-images"
    sessions_path = (
        Path(session_dir) if session_dir is not None else audio_path.parent / "practicas"
    )
    store = SessionStore(sessions_path)
    # S20: the chapter catalog comes from data/context/capitulos.json (built by
    # scripts/build_context.py). An empty catalog disables chapter scoping.
    chapter_catalog = list(chapters or [])
    # T11: one asyncio.Lock per practice session serializes every jury
    # mutation (start, direct start and answers), so two submissions can never
    # interleave between the "awaiting answer" check and the recorded turn.
    jury_locks: dict[str, asyncio.Lock] = {}

    def jury_lock(session: PracticeSession) -> asyncio.Lock:
        lock = jury_locks.get(session.id)
        if lock is None:
            lock = asyncio.Lock()
            jury_locks[session.id] = lock
        return lock

    # -- chapter scopes (S20) ----------------------------------------------

    def session_scope(session: PracticeSession) -> str:
        """Labels of the chosen chapters, empty when the whole thesis is chosen."""
        chosen = selected_chapters(session.jury_chapters, chapter_catalog)
        return ", ".join(chapter["label"] for chapter in chosen)

    def session_scope_block(session: PracticeSession) -> str:
        return chapter_scope_block(session.jury_chapters, chapter_catalog)

    def session_chapter_labels(session: PracticeSession) -> list[str]:
        return [
            chapter["label"]
            for chapter in selected_chapters(
                session.jury_chapters, chapter_catalog
            )
        ]

    def resolve_chapters(raw: list[str] | None) -> list[str]:
        if not raw:
            return []
        known = [chapter["id"] for chapter in chapter_catalog]
        unknown = [chapter_id for chapter_id in raw if chapter_id not in known]
        if unknown:
            valid = ", ".join(known) if known else "ninguno"
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Capítulo desconocido: {', '.join(unknown)}. "
                    f"Válidos: {valid}."
                ),
            )
        return list(dict.fromkeys(raw))

    def apply_chapter_scope(
        session: PracticeSession, chapter_ids: list[str]
    ) -> None:
        """Store the scope; a change invalidates every pending prefetch."""
        with store.lock:
            changed = list(chapter_ids) != list(session.jury_chapters)
            session.jury_chapters = list(chapter_ids)
        if changed:
            clear_prefetches(session.id)

    slide_definitions = load_slides(slides_path)

    def slide_image_version(number: int) -> int:
        definition = next(
            (item for item in slide_definitions if item["number"] == number), None
        )
        if definition is None:
            return 0
        try:
            return int((slides_path / definition["image"]).stat().st_mtime)
        except OSError:
            return 0

    def slide_image_url(number: int) -> str:
        # Version the URL with the file's mtime so a rebuilt deck is never
        # served from the browser cache under an unchanged address.
        return f"/api/slides/{number}/image?v={slide_image_version(number)}"
    picker = rng if rng is not None else random.Random()
    interjection_rate = min(
        1.0, max(0.0, float(interjection_probability))
    )
    grading_config = dict(DEFAULT_GRADING)
    if isinstance(grading, dict):
        grading_config.update(grading)
    grading_scale = grading_settings(grading_config)[0]

    deadline_config = dict(DEFAULT_DEADLINES)
    if isinstance(deadlines, dict):
        deadline_config.update(
            {str(key): float(value) for key, value in deadlines.items()}
        )

    app = FastAPI(title="Simulador de jurado", version="0.1.0")
    # Compress JSON and static text, never audio or images (already compressed,
    # and audio is streamed while it is still being synthesized).
    app.add_middleware(SelectiveGZip)
    app.state.brain = brain
    app.state.stt = stt
    app.state.tts = tts
    app.state.store = store
    app.state.slides = slide_definitions
    app.state.chapters = chapter_catalog

    # -- background work (S13) ---------------------------------------------

    background = executor or ThreadPoolExecutor(
        max_workers=6, thread_name_prefix="jurado-bg"
    )
    if executor is None:
        app.router.on_shutdown.append(
            lambda: background.shutdown(wait=False, cancel_futures=True)
        )
    # S15 deliberation runs after the last answer; tests can inject a
    # deterministic executor here without touching warm-up/prefetch.
    deliberation_background = deliberation_executor or background

    fillers_path = audio_path / "fillers"
    filler_lock = threading.Lock()

    def prepare_fillers_once() -> dict[str, list[str]]:
        """Reuse existing filler clips and synthesize only what is missing."""

        with filler_lock:
            return synthesize_fillers(
                tts, fillers_path, JUDGE_ROLES, filler_phrases
            )

    if prepare_fillers_on_startup and filler_phrases:
        app.router.on_startup.append(
            lambda: background.submit(prepare_fillers_once)
        )

    warmup_lock = threading.Lock()
    warmup_started: set[str] = set()
    jury_sessions_lock = threading.Lock()
    prefetch_lock = threading.Lock()
    prefetch_slots: dict[tuple[str, str], PrefetchedQuestion] = {}
    # S19: generated chunk files that a client may already be requesting.
    pending_audio: dict[str, Future] = {}
    pending_audio_lock = threading.Lock()
    # Brain asks run on their own pool so a timed-out turn never starves the
    # background executor (prefetch, fillers, warm-up).
    ask_pool = ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="jurado-ask"
    )
    app.router.on_shutdown.append(
        lambda: ask_pool.shutdown(wait=False, cancel_futures=True)
    )

    # -- serialization -----------------------------------------------------

    def slide_payload(slide: SlideState) -> dict[str, Any]:
        return {
            "number": slide.number,
            "text": slide.text,
            "image_url": slide_image_url(slide.number),
            "transcript": slide.transcript,
            "duration_seconds": slide.duration_seconds,
            "transcription_status": slide.transcription_status,
            "transcription_error": slide.transcription_error,
            "has_audio": bool(slide.audio),
            "feedback": slide.feedback,
            "attempts": list(slide.attempts),
        }

    def jury_payload(session: PracticeSession) -> dict[str, Any]:
        jury = session.jury
        return {
            "status": jury.status,
            "mode": session.jury_mode,
            "questions_per_judge": jury.questions_per_judge,
            "chapters": list(session.jury_chapters),
            "scope_label": (
                session_scope(session) or ALL_CHAPTERS_LABEL
            ),
            "current_judge": jury.current_judge,
            "current_role": jury.current_role,
            "turn_kind": jury.turn_kind,
            "current_question": jury.current_question,
            "current_turn_id": jury.current_turn_id,
            "current_audio_url": (
                f"/api/audio/{jury.current_audio}" if jury.current_audio else None
            ),
            "audio_chunks": [
                f"/api/audio/{name}" for name in jury.current_audio_chunks
            ],
            "timings": jury.last_timings,
            "greeting": jury.greeting,
            "judges": [
                {
                    "role": judge.role,
                    "name": judge.name,
                    "persona": judge.persona,
                    "asked": judge.asked,
                    "remaining": max(
                        0, jury.questions_per_judge - judge.asked
                    ),
                    "follow_ups": judge.follow_ups,
                    "interjections": judge.interjections,
                    "turns": [asdict(turn) for turn in judge.turns],
                    "verdict": judge.verdict,
                    "score": judge.score,
                    "active": (
                        index == jury.current_judge
                        and jury.status == JURY_AWAITING_ANSWER
                    ),
                }
                for index, judge in enumerate(jury.judges)
            ],
            "final": jury.final,
        }

    def current_slide_number(session: PracticeSession) -> int:
        for slide in session.slides:
            if not slide.audio:
                return slide.number
        return session.slides[-1].number if session.slides else 1

    def session_payload(session: PracticeSession) -> dict[str, Any]:
        return {
            "id": session.id,
            "created_at": session.created_at,
            "elapsed_seconds": round(session.elapsed_seconds(), 2),
            "exposition_status": session.exposition_status,
            "exposition_mode": session.exposition_mode,
            "jury_mode": session.jury_mode,
            "current_slide": current_slide_number(session),
            "slide_timeline": [asdict(mark) for mark in session.slide_timeline],
            "full_audio": bool(session.full_audio),
            "full_duration_seconds": session.full_duration_seconds,
            "targets": {
                "min_minutes": target_minutes_min,
                "max_minutes": target_minutes_max,
            },
            "slides": [slide_payload(slide) for slide in session.slides],
            "jury": jury_payload(session),
            "tutor": {
                "name": tutor_name,
                "messages": [asdict(message) for message in session.tutor_messages],
            },
            "global": session.global_report,
            "saved_path": session.saved_path,
        }

    def require_session() -> PracticeSession:
        try:
            return store.require()
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail="No hay una práctica en curso."
            ) from exc

    def find_slide(session: PracticeSession, number: int) -> SlideState | None:
        return next(
            (slide for slide in session.slides if slide.number == number), None
        )

    def safe_save(session: PracticeSession) -> None:
        try:
            store.save(session)
        except Exception:  # pragma: no cover - disk failures are environment-specific
            logger.exception("Could not save practice session %s", session.id)

    # -- speech helpers ----------------------------------------------------

    def speak(text: str, role: str) -> str | None:
        try:
            result = tts.synthesize(text, role, audio_path)
        except Exception:
            logger.exception("TTS failed for role %s", role)
            return None
        if not result:
            return None
        return Path(result).name

    def transcribe_slide(
        session: PracticeSession, slide: SlideState, source: Path
    ) -> None:
        wav = _convert_to_wav(source)
        try:
            text = (stt.transcribe_audio(str(wav)) or "").strip()
        except Exception as exc:
            logger.exception("STT failed for slide %s", slide.number)
            with store.lock:
                slide.transcription_status = TRANSCRIPTION_ERROR
                slide.transcription_error = str(exc)
            if session.exposition_status == EXPOSITION_FINISHED:
                safe_save(session)
            return
        with store.lock:
            slide.transcript = text or None
            slide.transcription_status = (
                TRANSCRIPTION_DONE if text else TRANSCRIPTION_EMPTY
            )
            slide.transcription_error = None
        if session.exposition_status == EXPOSITION_FINISHED:
            safe_save(session)

    async def transcribe_upload(upload: UploadFile, prefix: str) -> str:
        suffix = Path(upload.filename or "").suffix or ".webm"
        name = f"{prefix}-{int(time.time() * 1000)}{suffix}"
        target = audio_path / name
        data = await upload.read()
        if not data:
            raise HTTPException(
                status_code=422, detail="El audio llegó vacío. Intenta de nuevo."
            )
        target.write_bytes(data)
        wav = await run_in_threadpool(_convert_to_wav, target)
        try:
            text = await run_in_threadpool(stt.transcribe_audio, str(wav))
        except Exception as exc:
            logger.exception("STT failed for %s", name)
            raise HTTPException(
                status_code=502,
                detail=f"No se pudo transcribir el audio: {exc}",
            ) from exc
        text = (text or "").strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail="No se entendió tu respuesta. Repite, por favor.",
            )
        return text

    def transcribe_full(
        session: PracticeSession,
        source: Path,
        marks: list[SlideMark],
        total_ms: int | None,
    ) -> None:
        """Transcribe the whole defense once and split it across the slides."""
        try:
            segments = stt.transcribe_segments(str(source))
        except Exception as exc:
            logger.exception("STT failed for the full defense recording")
            visited = {mark.slide for mark in marks}
            with store.lock:
                for slide in session.slides:
                    if slide.number in visited:
                        slide.transcription_status = TRANSCRIPTION_ERROR
                        slide.transcription_error = str(exc)
            if session.exposition_status == EXPOSITION_FINISHED:
                safe_save(session)
            return
        with store.lock:
            apply_full_transcription(session, marks, segments, total_ms)
        if session.exposition_status == EXPOSITION_FINISHED:
            safe_save(session)

    # -- brain helpers -----------------------------------------------------

    def deadline_for(kind: str) -> float:
        """Seconds a judge turn may think before it is interrupted."""
        return max(
            0.05,
            float(deadline_config.get(kind, DEFAULT_DEADLINES.get(kind, 25.0))),
        )

    def mark_deadline_notice(jury) -> None:
        jury.notice = BRAIN_DEADLINE_NOTICE

    def take_notice(session: PracticeSession) -> str | None:
        notice = session.jury.notice
        session.jury.notice = None
        return notice

    def timing_payload(judge, kind, stt_ms, timing, prefetched):
        return {
            "role": judge.role,
            "kind": kind,
            "stt_ms": round(float(stt_ms or 0.0), 1),
            "brain_ms": round(float((timing or {}).get("brain_ms") or 0.0), 1),
            "tts_ms": round(float((timing or {}).get("tts_ms") or 0.0), 1),
            "first_audio_ms": round(
                float((timing or {}).get("first_audio_ms") or 0.0), 1
            ),
            "wait_ms": round(float((timing or {}).get("wait_ms") or 0.0), 1),
            "prefetched": bool(prefetched),
        }

    def log_timings(timings: dict[str, Any]) -> None:
        logger.info(
            "jury turn role=%s kind=%s stt_ms=%.0f brain_ms=%.0f tts_ms=%.0f "
            "first_audio_ms=%.0f wait_ms=%.0f prefetched=%s",
            timings["role"],
            timings["kind"],
            timings["stt_ms"],
            timings["brain_ms"],
            timings["tts_ms"],
            timings["first_audio_ms"],
            timings["wait_ms"],
            timings["prefetched"],
        )

    def finalize_timings(session: PracticeSession, request_started: float) -> None:
        """Stamp the answer-to-response wait and log the turn."""
        timings = session.jury.last_timings
        if not isinstance(timings, dict):
            return
        timings["wait_ms"] = round(
            max(0.0, time.monotonic() - request_started) * 1000.0, 1
        )
        log_timings(timings)

    def interrupt_brain_session(session_id: str | None) -> bool:
        """Stop a stalled OpenCode turn so the session is idle again."""
        if not session_id:
            return False
        try:
            interrupt = getattr(brain, "interrupt", None)
            if callable(interrupt):
                interrupt(session_id)
            else:
                # app/brain.py exposes the authenticated HTTP client only.
                brain._request(  # noqa: SLF001 - no public interrupt helper yet
                    "POST", f"/api/session/{session_id}/interrupt"
                )
        except Exception:
            logger.warning(
                "Could not interrupt brain session %s", session_id, exc_info=True
            )
            return False
        return True

    def remaining_budget(budget_at: float | None) -> float | None:
        if budget_at is None:
            return None
        return max(0.05, budget_at - time.monotonic())

    def ask_brain(
        judge, kind: str, prompt: str, budget_at: float | None = None
    ) -> str:
        """Ask one judge with a hard per-turn deadline."""
        session_id = judge.brain_session_id
        if not session_id:
            raise RuntimeError(f"role {judge.role} has no brain session")
        deadline = deadline_for(kind)
        remaining = remaining_budget(budget_at)
        if remaining is not None:
            deadline = min(deadline, remaining)
        future = ask_pool.submit(brain.ask, session_id, prompt)
        started = time.monotonic()
        try:
            return future.result(timeout=deadline)
        except FuturesTimeout as exc:
            elapsed = time.monotonic() - started
            logger.warning(
                "brain deadline role=%s kind=%s after=%.1fs action=interrupt",
                judge.role,
                kind,
                elapsed,
            )
            interrupt_brain_session(session_id)
            raise BrainDeadline(judge.role, kind, elapsed) from exc

    def ask_judge(
        judge,
        kind: str,
        prompt: str,
        allow_retry: bool = True,
        budget_at: float | None = None,
    ) -> str:
        """Ask with one retry in a fresh session when the deadline expires."""
        try:
            return ask_brain(judge, kind, prompt, budget_at)
        except BrainDeadline as exc:
            if not allow_retry:
                raise
            fresh = brain.create_session(judge.role)
            with jury_sessions_lock:
                judge.brain_session_id = fresh
            logger.warning(
                "brain deadline role=%s kind=%s after=%.1fs "
                "action=retry_fresh_session session=%s",
                exc.role,
                exc.kind,
                exc.seconds,
                fresh,
            )
            return ask_brain(judge, kind, prompt, budget_at)

    def audio_ready(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    def chunk_plan(role: str, text: str) -> list[AudioChunk]:
        """Split a question into sentence chunks with pre-allocated file names."""
        pieces = split_speech_chunks(text) or ([text.strip()] if text.strip() else [])
        return [
            AudioChunk(
                text=piece,
                target=audio_path / f"{_audio_slug(role)}-{uuid.uuid4().hex}.wav",
            )
            for piece in pieces
        ]

    def track_chunk(chunk: AudioChunk, role: str, priority: int) -> Future:
        """Queue a chunk and expose it to waiting audio GETs."""
        try:
            future = tts.request_synthesis(
                chunk.text, role, audio_path, chunk.target, priority=priority
            )
        except Exception as exc:
            future = Future()
            future.set_exception(exc)
            return future
        name = chunk.target.name
        with pending_audio_lock:
            pending_audio[name] = future

        def cleanup(done: Future) -> None:
            with pending_audio_lock:
                if pending_audio.get(name) is done:
                    pending_audio.pop(name, None)

        future.add_done_callback(cleanup)
        return future

    def watch_first_audio(
        timings: dict[str, Any],
        speech: TurnSpeech,
        request_started: float,
        role: str,
        kind: str,
    ) -> None:
        """Record first_audio_ms as soon as chunk 1 is on disk."""
        if speech.first_ready is None:
            return

        def done(completed: Future) -> None:
            if completed.cancelled() or completed.exception() is not None:
                return
            ready_at = time.monotonic()
            first_ms = round(
                max(0.0, (ready_at - request_started) * 1000.0), 1
            )
            tts_ms = round(
                max(0.0, (ready_at - speech.tts_started) * 1000.0), 1
            )
            with store.lock:
                timings["first_audio_ms"] = first_ms
                timings["tts_ms"] = tts_ms
            logger.info(
                "jury turn audio role=%s kind=%s first_audio_ms=%.0f tts_ms=%.0f",
                role,
                kind,
                first_ms,
                tts_ms,
            )

        speech.first_ready.add_done_callback(done)

    def generate_speech(judge, question: str) -> TurnSpeech:
        """Queue every chunk of a live question, first chunk first."""
        plan = chunk_plan(judge.role, question)
        tts_started = time.monotonic()
        first_ready: Future | None = None
        for index, chunk in enumerate(plan):
            future = track_chunk(chunk, judge.role, PRIORITY_HIGH)
            if index == 0:
                first_ready = future
        return TurnSpeech(
            question=question,
            chunks=[chunk.target.name for chunk in plan],
            timing={"brain_ms": 0.0, "first_audio_ms": 0.0},
            first_ready=first_ready,
            tts_started=tts_started,
        )

    def generate_question_sync(
        judge,
        prompt: str,
        kind: str,
        allow_retry: bool,
        budget_at: float | None = None,
    ) -> TurnSpeech:
        """Brain reply first, then the streamed audio plan (never blocks on TTS)."""
        started = time.monotonic()
        reply = ask_judge(
            judge, kind, prompt, allow_retry=allow_retry, budget_at=budget_at
        )
        question = _clean_reply(reply or "")
        brain_ms = (time.monotonic() - started) * 1000.0
        if not question:
            raise RuntimeError("the judge returned an empty question")
        speech = generate_speech(judge, question)
        speech.timing["brain_ms"] = brain_ms
        return speech

    async def generate_question(
        judge,
        prompt: str,
        kind: str,
        allow_retry: bool = True,
        budget_at: float | None = None,
    ) -> TurnSpeech:
        try:
            return await run_in_threadpool(
                generate_question_sync,
                judge,
                prompt,
                kind,
                allow_retry,
                budget_at,
            )
        except BrainDeadline:
            raise
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Brain failed for role %s", judge.role)
            raise HTTPException(
                status_code=502,
                detail=f"El juez no respondió a tiempo: {exc}",
            ) from exc

    def build_prefetched_question(
        judge, number: int, prompt: str, slot: PrefetchedQuestion
    ) -> None:
        """Store the text immediately and queue its audio at low priority."""
        slot.started = True
        started = time.monotonic()
        try:
            reply = ask_judge(
                judge, JURY_TURN_QUESTION, prompt, allow_retry=False
            )
            question = _clean_reply(reply or "")
            if not question:
                raise RuntimeError("the judge returned an empty question")
            slot.brain_ms = (time.monotonic() - started) * 1000.0
            slot.question = question
            slot.plan = chunk_plan(judge.role, question)
            for chunk in slot.plan:
                track_chunk(chunk, judge.role, PRIORITY_LOW)
        except Exception as exc:  # noqa: BLE001 - the consumer falls back live
            slot.error = exc
        finally:
            slot.text_ready.set()

    def remaining_for(jury, judge) -> int:
        return max(0, jury.questions_per_judge - judge.asked)

    def ensure_judge_sessions(session: PracticeSession) -> list[str | None]:
        """Create the missing judge sessions, reusing the existing ones."""

        with jury_sessions_lock:
            missing = [
                judge
                for judge in session.jury.judges
                if not judge.brain_session_id
            ]
            for judge in missing:
                session_id = brain.create_session(judge.role)
                with store.lock:
                    if not judge.brain_session_id:
                        judge.brain_session_id = session_id
            return [judge.brain_session_id for judge in session.jury.judges]

    def warm_up_judge(judge) -> None:
        if not judge.brain_session_id:
            return
        started = time.monotonic()
        try:
            ask_judge(
                judge, JURY_TURN_QUESTION, WARMUP_PROMPT, allow_retry=False
            )
        except BrainDeadline as exc:
            logger.warning(
                "brain deadline role=%s kind=%s after=%.1fs action=skip_warmup",
                exc.role,
                exc.kind,
                exc.seconds,
            )
            return
        except Exception:
            logger.warning(
                "Jury warm-up failed for role %s", judge.role, exc_info=True
            )
            return
        logger.info(
            "Jury warm-up role=%s ms=%.0f",
            judge.role,
            (time.monotonic() - started) * 1000.0,
        )

    def warm_up_jury(session: PracticeSession) -> None:
        """Pay each judge session's cold start in parallel."""
        try:
            ensure_judge_sessions(session)
        except Exception:
            logger.exception("Jury warm-up could not create judge sessions")
            return
        judges = [
            judge for judge in session.jury.judges if judge.brain_session_id
        ]
        if not judges:
            return
        with ThreadPoolExecutor(
            max_workers=len(judges), thread_name_prefix="jurado-warmup"
        ) as pool:
            list(pool.map(warm_up_judge, judges))

    def warm_up_now(session: PracticeSession) -> None:
        """Warm up synchronously; direct mode needs it before the first ask."""
        with warmup_lock:
            warmup_started.add(session.id)
        warm_up_jury(session)

    def schedule_warmup(session: PracticeSession) -> None:
        if not warmup_enabled:
            return
        with warmup_lock:
            if session.id in warmup_started:
                return
            warmup_started.add(session.id)
        background.submit(warm_up_jury, session)

    def prefetch_key(
        session: PracticeSession, judge
    ) -> tuple[str, str, str]:
        """Include the chapter scope so another scope is never served."""
        scope_key = ",".join(sorted(set(session.jury_chapters))) or "all"
        return (session.id, judge.role, scope_key)

    def schedule_prefetch(
        session: PracticeSession, judge, number: int, prompt: str
    ) -> None:
        if not prefetch_enabled or not judge.brain_session_id:
            return
        key = prefetch_key(session, judge)
        slot = PrefetchedQuestion(
            number=number, history_size=answered_turns(session)
        )
        with prefetch_lock:
            if key in prefetch_slots:
                return
            prefetch_slots[key] = slot
        try:
            background.submit(
                build_prefetched_question, judge, number, prompt, slot
            )
        except Exception as exc:  # noqa: BLE001 - never leave a waiter hanging
            slot.error = exc
            slot.text_ready.set()
        logger.info(
            "jury prefetch scheduled role=%s number=%s", judge.role, number
        )

    async def acquire_prefetch(
        session: PracticeSession, judge, number: int
    ) -> PrefetchedQuestion | None:
        """Take the prefetched question for this number, if there is one."""

        key = prefetch_key(session, judge)
        with prefetch_lock:
            slot = prefetch_slots.pop(key, None)
        if slot is None:
            return None
        if slot.number != number or slot.history_size != answered_turns(session):
            # A stale slot (another number, or written before the student's
            # latest answers) must free the judge session before we ask again.
            try:
                await run_in_threadpool(slot.wait_text)
            except Exception:
                pass
            return None
        return slot

    async def settle_prefetch(session: PracticeSession, judge) -> None:
        """Wait for an in-flight prefetch without consuming it."""

        key = prefetch_key(session, judge)
        with prefetch_lock:
            slot = prefetch_slots.get(key)
        if slot is None or not slot.started:
            return
        try:
            await run_in_threadpool(slot.wait_text)
        except Exception:
            pass

    def clear_prefetches(session_id: str | None = None) -> None:
        with prefetch_lock:
            keys = [
                key
                for key in prefetch_slots
                if session_id is None or key[0] == session_id
            ]
            for key in keys:
                del prefetch_slots[key]

    def opening_question_prompt(session: PracticeSession) -> str:
        """First question prompt: from the transcript, or direct-mode fallback."""
        scope = session_scope_block(session)
        history = room_history_block(session)
        if (
            session.jury_mode == JURY_MODE_DIRECT
            or session.exposition_words() <= 0
        ):
            return direct_first_question_prompt(scope, history)
        return first_question_prompt(session.exposition_transcript(), scope, history)

    def schedule_first_question_prefetches(
        session: PracticeSession, first_index: int | None = None
    ) -> None:
        """Warm the first asker's opening question right after the exposition.

        S23: only the judge who will open is prefetched. Any other judge's
        question would be written before the student's first answers and could
        repeat them, so it is generated live with the room history instead.
        """

        if not prefetch_enabled:
            return
        if (
            session.jury_mode != JURY_MODE_DIRECT
            and session.exposition_words() <= 0
        ):
            return
        try:
            ensure_judge_sessions(session)
        except Exception:
            logger.exception("Could not create judge sessions for prefetch")
            return
        jury = session.jury
        if first_index is None:
            first_index = pick_next_main_index(jury)
        if first_index is None:
            return
        judge = jury.judges[first_index]
        if judge.asked != 0 or remaining_for(jury, judge) <= 0:
            return
        jury.planned_first = first_index
        schedule_prefetch(session, judge, 1, opening_question_prompt(session))

    def pick_next_main_index(
        jury, avoid: int | None = None
    ) -> int | None:
        """Random judge with main budget left, never the same one twice."""
        candidates = [
            index
            for index, judge in enumerate(jury.judges)
            if remaining_for(jury, judge) > 0
        ]
        if not candidates:
            return None
        if avoid is not None and len(candidates) > 1:
            different = [index for index in candidates if index != avoid]
            if different:
                candidates = different
        return picker.choice(candidates)

    def pick_interjection_index(jury, asker_index: int) -> int:
        candidates = [
            index
            for index in range(len(jury.judges))
            if index != asker_index
        ]
        return picker.choice(candidates)

    def should_follow_up(jury, judge) -> bool:
        # Same single follow-up as before, on top of the new main-question budget.
        return jury.questions_per_judge >= 2 and judge.follow_ups == 0

    async def take_main_question(
        session: PracticeSession,
        judge,
        number: int,
        prompt: str,
        request_started: float,
        budget_at: float | None = None,
        allow_retry: bool = True,
    ) -> tuple[TurnSpeech, bool]:
        """Consume the prefetched question or generate it live (streamed audio)."""
        slot = await acquire_prefetch(session, judge, number)
        if slot is not None:
            wait_started = time.monotonic()
            try:
                remaining = remaining_budget(budget_at)
                if remaining is None:
                    remaining = deadline_for(JURY_TURN_QUESTION) + 5.0
                question = await run_in_threadpool(
                    slot.wait_text_bounded, remaining
                )
                # Audio may already exist (prefetch finished) or still be
                # queued; promoting it to the live priority never re-synthesizes.
                ready_before = bool(
                    slot.plan and audio_ready(slot.plan[0].target)
                )
                tts_started = time.monotonic()
                first_ready: Future | None = None
                for index, chunk in enumerate(slot.plan):
                    future = track_chunk(chunk, judge.role, PRIORITY_HIGH)
                    if index == 0:
                        first_ready = future
                if ready_before:
                    first_ready = None
                return (
                    TurnSpeech(
                        question=question,
                        chunks=[chunk.target.name for chunk in slot.plan],
                        timing={
                            "brain_ms": slot.brain_ms,
                            "first_audio_ms": 0.0,
                        },
                        first_ready=first_ready,
                        tts_started=tts_started,
                    ),
                    True,
                )
            except PrefetchTimeout as exc:
                # The background ask is still using that session: hand the turn
                # to another judge instead of crossing prompts in it.
                logger.warning(
                    "brain deadline role=%s kind=%s after=%.1fs "
                    "action=skip_prefetch",
                    judge.role,
                    JURY_TURN_QUESTION,
                    time.monotonic() - wait_started,
                )
                raise BrainDeadline(
                    judge.role, JURY_TURN_QUESTION, 0.0
                ) from exc
            except Exception:
                logger.warning(
                    "Prefetched question failed for role %s; generating live",
                    judge.role,
                    exc_info=True,
                )
        speech = await generate_question(
            judge,
            prompt,
            JURY_TURN_QUESTION,
            allow_retry=allow_retry,
            budget_at=budget_at,
        )
        return speech, False

    async def ask_main_question(
        session: PracticeSession,
        judge_index: int,
        stt_ms: float = 0.0,
        request_started: float | None = None,
        allow_alternative: bool = True,
    ) -> int | None:
        """Ask the next main question; return the judge that asked, or None."""
        if request_started is None:
            request_started = time.monotonic()
        budget_at = (
            request_started + deadline_for(JURY_TURN_QUESTION) * 2.0 + 5.0
        )
        jury = session.jury
        judge = jury.judges[judge_index]
        number = judge.asked + 1
        prompt = (
            opening_question_prompt(session)
            if judge.asked == 0
            else next_main_question_prompt(
                session_scope_block(session), room_history_block(session)
            )
        )
        prefetched = False
        speech: TurnSpeech | None = None
        try:
            speech, prefetched = await take_main_question(
                session, judge, number, prompt, request_started, budget_at
            )
        except BrainDeadline as exc:
            logger.warning(
                "brain deadline role=%s kind=%s after=%.1fs action=next_judge",
                exc.role,
                exc.kind,
                exc.seconds,
            )
            mark_deadline_notice(jury)
        if speech is None and allow_alternative:
            alternative = pick_next_main_index(jury, avoid=judge_index)
            if alternative is not None and alternative != judge_index:
                alt = jury.judges[alternative]
                alt_number = alt.asked + 1
                alt_prompt = (
                    opening_question_prompt(session)
                    if alt.asked == 0
                    else next_main_question_prompt(
                        session_scope_block(session), room_history_block(session)
                    )
                )
                try:
                    speech, prefetched = await take_main_question(
                        session,
                        alt,
                        alt_number,
                        alt_prompt,
                        request_started,
                        budget_at,
                        allow_retry=False,
                    )
                    judge_index, judge = alternative, alt
                except BrainDeadline as exc:
                    logger.warning(
                        "brain deadline role=%s kind=%s after=%.1fs "
                        "action=skip_question",
                        exc.role,
                        exc.kind,
                        exc.seconds,
                    )
                    speech = None
        if speech is None:
            mark_deadline_notice(jury)
            return None
        timings = timing_payload(
            judge, JURY_TURN_QUESTION, stt_ms, speech.timing, prefetched
        )
        with store.lock:
            judge.asked += 1
            jury.current_judge = judge_index
            jury.current_role = judge.role
            jury.turn_kind = JURY_TURN_QUESTION
            jury.current_question = speech.question
            jury.current_audio = speech.chunks[0] if speech.chunks else None
            jury.current_audio_chunks = list(speech.chunks)
            jury.last_main_judge = judge_index
            jury.status = JURY_AWAITING_ANSWER
            jury.last_timings = timings
            jury.begin_turn()
        watch_first_audio(
            timings, speech, request_started, judge.role, JURY_TURN_QUESTION
        )
        return judge_index

    async def ask_follow_up(
        session: PracticeSession,
        judge_index: int,
        answer: str,
        stt_ms: float = 0.0,
        request_started: float | None = None,
    ) -> bool:
        if request_started is None:
            request_started = time.monotonic()
        budget_at = (
            request_started + deadline_for(JURY_TURN_FOLLOW_UP) * 2.0 + 5.0
        )
        jury = session.jury
        judge = jury.judges[judge_index]
        await settle_prefetch(session, judge)
        try:
            speech = await generate_question(
                judge,
                follow_up_prompt(
                    answer, session_scope_block(session), room_history_block(session)
                ),
                JURY_TURN_FOLLOW_UP,
                budget_at=budget_at,
            )
        except BrainDeadline as exc:
            logger.warning(
                "brain deadline role=%s kind=%s after=%.1fs action=skip_follow_up",
                exc.role,
                exc.kind,
                exc.seconds,
            )
            mark_deadline_notice(jury)
            return False
        timings = timing_payload(
            judge, JURY_TURN_FOLLOW_UP, stt_ms, speech.timing, False
        )
        with store.lock:
            judge.follow_ups += 1
            jury.current_judge = judge_index
            jury.current_role = judge.role
            jury.turn_kind = JURY_TURN_FOLLOW_UP
            jury.current_question = speech.question
            jury.current_audio = speech.chunks[0] if speech.chunks else None
            jury.current_audio_chunks = list(speech.chunks)
            jury.status = JURY_AWAITING_ANSWER
            jury.last_timings = timings
            jury.begin_turn()
        watch_first_audio(
            timings, speech, request_started, judge.role, JURY_TURN_FOLLOW_UP
        )
        return True

    async def ask_interjection(
        session: PracticeSession,
        judge_index: int,
        colleague: str,
        question: str,
        answer: str,
        stt_ms: float = 0.0,
        request_started: float | None = None,
    ) -> bool:
        if request_started is None:
            request_started = time.monotonic()
        budget_at = (
            request_started + deadline_for(JURY_TURN_INTERJECTION) * 2.0 + 5.0
        )
        jury = session.jury
        judge = jury.judges[judge_index]
        prompt = interjection_prompt(
            colleague,
            question,
            answer,
            session_scope_block(session),
            room_history_block(session),
        )
        # An answer-dependent turn stays live and never shares the session
        # with an in-flight prefetch.
        await settle_prefetch(session, judge)
        try:
            speech = await generate_question(
                judge,
                prompt,
                JURY_TURN_INTERJECTION,
                allow_retry=False,
                budget_at=budget_at,
            )
        except BrainDeadline as exc:
            logger.warning(
                "brain deadline role=%s kind=%s after=%.1fs "
                "action=skip_interjection",
                exc.role,
                exc.kind,
                exc.seconds,
            )
            mark_deadline_notice(jury)
            return False
        timings = timing_payload(
            judge, JURY_TURN_INTERJECTION, stt_ms, speech.timing, False
        )
        with store.lock:
            judge.interjections += 1
            jury.current_judge = judge_index
            jury.current_role = judge.role
            jury.turn_kind = JURY_TURN_INTERJECTION
            jury.current_question = speech.question
            jury.current_audio = speech.chunks[0] if speech.chunks else None
            jury.current_audio_chunks = list(speech.chunks)
            jury.status = JURY_AWAITING_ANSWER
            jury.last_timings = timings
            jury.begin_turn()
        watch_first_audio(
            timings, speech, request_started, judge.role, JURY_TURN_INTERJECTION
        )
        return True

    def evaluate_judge(
        session: PracticeSession, judge
    ) -> dict[str, Any]:
        """One judge grades the whole defense (runs inside the parallel pool)."""
        try:
            raw = ask_judge(
                judge,
                "veredicto",
                deliberation_prompt(
                    session,
                    judge,
                    target_minutes_min,
                    target_minutes_max,
                    grading_scale,
                    scope_label=session_scope(session),
                ),
            )
        except BrainDeadline as exc:
            logger.warning(
                "brain deadline role=%s kind=%s after=%.1fs "
                "action=grade_without_judge",
                exc.role,
                exc.kind,
                exc.seconds,
            )
            parsed = parse_deliberation_reply("", grading_scale)
            parsed["comentario"] = (
                "El juez no respondió durante la deliberación."
            )
        except Exception:
            logger.exception("Deliberation failed for role %s", judge.role)
            parsed = parse_deliberation_reply("", grading_scale)
            parsed["comentario"] = (
                "El juez no respondió durante la deliberación."
            )
        else:
            parsed = parse_deliberation_reply(raw or "", grading_scale)
        parsed["role"] = judge.role
        parsed["name"] = judge.name
        return parsed

    def collect_deliberation(session: PracticeSession) -> None:
        """Run the three judges in parallel, grade in Python and close (S15)."""
        jury = session.jury
        try:
            judges = list(jury.judges)
            with ThreadPoolExecutor(
                max_workers=max(1, len(judges)),
                thread_name_prefix="jurado-deliberation",
            ) as pool:
                futures = [
                    pool.submit(evaluate_judge, session, judge)
                    for judge in judges
                ]
                evaluations = [future.result() for future in futures]

            if session.jury_mode == JURY_MODE_DIRECT:
                # Direct mode never grades an exposition, whatever the model
                # returns: the nota is the answer score alone.
                for evaluation in evaluations:
                    evaluation["exposicion"] = None

            grade = compute_jury_grade(evaluations, grading_config)
            final = {
                **grade,
                "mode": session.jury_mode,
                "chapters": session_chapter_labels(session),
                "verdicts": grade["judges"],
                "questions": deliberation_questions(session),
            }
            final["closing"] = build_closing_line(final)
            audio = speak(final["closing"], "juez_presidente")
            final["closing_audio_url"] = (
                f"/api/audio/{audio}" if audio else None
            )

            with store.lock:
                for judge, row in zip(judges, grade["judges"]):
                    judge.verdict = (row.get("comentario") or "").strip() or None
                    judge.score = row.get("nota")
                jury.status = JURY_DONE
                jury.current_question = None
                jury.current_audio = None
                jury.current_audio_chunks = []
                jury.current_role = None
                jury.turn_kind = None
                jury.current_turn_id = None
                jury.last_timings = None
                jury.final = final
            safe_save(session)
            logger.info(
                "jury deliberation final_grade=%s result=%s mention=%s",
                final.get("final_grade"),
                final.get("result"),
                final.get("mention"),
            )
        except Exception:
            logger.exception("Jury deliberation failed")
            fallback = compute_jury_grade([], grading_config)
            with store.lock:
                jury.status = JURY_DONE
                jury.current_question = None
                jury.current_audio = None
                jury.current_audio_chunks = []
                jury.current_role = None
                jury.turn_kind = None
                jury.current_turn_id = None
                jury.last_timings = None
                jury.final = {
                    **fallback,
                    "mode": session.jury_mode,
                    "chapters": session_chapter_labels(session),
                    "closing": "",
                    "closing_audio_url": None,
                    "questions": deliberation_questions(session),
                }
            safe_save(session)

    # -- slides ------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "slides": len(slide_definitions)}

    @app.get("/api/chapters")
    async def list_chapters() -> dict[str, Any]:
        """Chapter scopes for the jury selector (S20)."""
        return {
            "chapters": [
                {
                    "id": chapter["id"],
                    "label": chapter["label"],
                    "start_line": chapter.get("start_line"),
                    "end_line": chapter.get("end_line"),
                }
                for chapter in chapter_catalog
            ]
        }

    @app.get("/api/slides")
    async def list_slides() -> dict[str, Any]:
        return {
            "slides": [
                {
                    "number": definition["number"],
                    "text": definition["text"],
                    "image_url": slide_image_url(definition["number"]),
                }
                for definition in slide_definitions
            ]
        }

    @app.get("/api/slides/{number}/image")
    async def slide_image(number: int, v: int | None = None):
        definition = next(
            (item for item in slide_definitions if item["number"] == number), None
        )
        if definition is None:
            raise HTTPException(status_code=404, detail="Lámina no encontrada.")
        path = slides_path / definition["image"]
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Imagen no encontrada.")
        # The URL carries the file's mtime: the current version never changes,
        # so the browser keeps it; any other version revalidates.
        if v is not None and v == slide_image_version(number):
            cache = "public, max-age=31536000, immutable"
        else:
            cache = "no-cache"
        return FileResponse(path, headers={"Cache-Control": cache})

    # -- session -----------------------------------------------------------

    @app.post("/api/session")
    async def create_session(payload: NewSessionRequest | None = None) -> dict[str, Any]:
        questions = payload.questions_per_judge if payload else 2
        if not 1 <= questions <= MAX_QUESTIONS_PER_JUDGE:
            raise HTTPException(
                status_code=422,
                detail="Las preguntas por juez deben ser 1 o 2.",
            )
        session = store.start(slide_definitions, questions_per_judge=questions)
        jury_locks.clear()
        clear_prefetches()
        return session_payload(session)

    @app.get("/api/session")
    async def get_session() -> dict[str, Any]:
        return session_payload(require_session())

    @app.post("/api/exposition/start")
    async def start_exposition(
        payload: ExpositionStartRequest | None = None,
    ) -> dict[str, Any]:
        session = require_session()
        mode = payload.mode if payload else None
        if mode is not None and mode not in (
            EXPOSITION_MODE_PRACTICE,
            EXPOSITION_MODE_REAL,
        ):
            raise HTTPException(
                status_code=422,
                detail="El modo debe ser 'practice' o 'real'.",
            )
        with store.lock:
            if mode is not None:
                session.exposition_mode = mode
            if session.exposition_started_at is None:
                session.exposition_started_at = time.time()
            session.exposition_status = EXPOSITION_RUNNING
        schedule_warmup(session)
        return session_payload(session)

    @app.post("/api/exposition/opening")
    async def exposition_opening() -> dict[str, Any]:
        """Speak the president's opening line before the real defense starts."""
        require_session()
        text = render_opening_line(opening_line, student_name)
        audio = await run_in_threadpool(speak, text, "juez_presidente")
        return {
            "text": text,
            "audio_url": f"/api/audio/{audio}" if audio else None,
        }

    @app.post("/api/session/finish")
    async def finish_session() -> dict[str, Any]:
        session = require_session()
        with store.lock:
            session.exposition_status = EXPOSITION_FINISHED
            if session.exposition_finished_at is None:
                session.exposition_finished_at = time.time()
        path = store.save(session)
        await run_in_threadpool(schedule_first_question_prefetches, session)
        return {"session": session_payload(session), "saved_path": str(path)}

    # -- exposition --------------------------------------------------------

    @app.post("/api/exposition/full/audio")
    async def upload_full_audio(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        timeline: str = Form("[]"),
        duration: float | None = Form(None),
    ) -> dict[str, Any]:
        """Store one continuous defense recording plus its slide timeline."""
        session = require_session()
        try:
            marks = parse_timeline(
                timeline, {slide.number for slide in session.slides}
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        suffix = Path(file.filename or "").suffix or ".webm"
        name = f"defensa-{session.id}-{int(time.time() * 1000)}{suffix}"
        target = audio_path / name
        data = await file.read()
        if not data:
            raise HTTPException(
                status_code=422, detail="El audio llegó vacío. Intenta de nuevo."
            )
        target.write_bytes(data)
        wav = await run_in_threadpool(_convert_to_wav, target)
        total_ms: int | None = None
        if duration is not None and duration > 0:
            total_ms = int(round(float(duration) * 1000))
        else:
            wav_seconds = await run_in_threadpool(wav_duration_seconds, wav)
            if wav_seconds:
                total_ms = int(round(wav_seconds * 1000))
        visited = {mark.slide for mark in marks}
        with store.lock:
            session.exposition_mode = EXPOSITION_MODE_REAL
            session.slide_timeline = marks
            session.full_audio = name
            session.full_duration_seconds = (
                round(total_ms / 1000.0, 2) if total_ms else None
            )
            for slide in session.slides:
                if slide.number not in visited:
                    continue
                slide.audio = name
                slide.transcript = None
                slide.duration_seconds = None
                slide.transcription_status = TRANSCRIPTION_RUNNING
                slide.transcription_error = None
        background_tasks.add_task(transcribe_full, session, wav, marks, total_ms)
        return session_payload(session)

    @app.get("/api/exposition/{number}/status")
    async def slide_status(number: int) -> dict[str, Any]:
        # Light poll for the practice tab instead of the whole session.
        session = require_session()
        slide = find_slide(session, number)
        if slide is None:
            raise HTTPException(status_code=404, detail="Lámina no encontrada.")
        with store.lock:
            return {
                "number": slide.number,
                "transcription_status": slide.transcription_status,
                "transcription_error": slide.transcription_error,
                "transcript": slide.transcript,
                "duration_seconds": slide.duration_seconds,
                "has_audio": bool(slide.audio),
            }

    @app.post("/api/exposition/{number}/audio")
    async def upload_slide_audio(
        number: int,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        duration: float | None = Form(None),
    ) -> dict[str, Any]:
        session = require_session()
        slide = find_slide(session, number)
        if slide is None:
            raise HTTPException(status_code=404, detail="Lámina no encontrada.")
        suffix = Path(file.filename or "").suffix or ".webm"
        name = f"slide-{slide.number:02d}-{session.id}-{int(time.time() * 1000)}{suffix}"
        target = audio_path / name
        data = await file.read()
        if not data:
            raise HTTPException(
                status_code=422, detail="El audio llegó vacío. Intenta de nuevo."
            )
        target.write_bytes(data)
        with store.lock:
            slide.audio = name
            slide.transcription_status = TRANSCRIPTION_RUNNING
            slide.transcription_error = None
            # A new take makes the previous feedback stale; keep it as an
            # attempt so the evaluator can compare the retake with it.
            slide.archive_feedback()
            if duration is not None and duration >= 0:
                slide.duration_seconds = round(float(duration), 2)
        background_tasks.add_task(transcribe_slide, session, slide, target)
        return {
            "slide": slide.number,
            "transcription_status": slide.transcription_status,
            "has_audio": True,
        }

    # -- jury --------------------------------------------------------------

    @app.post("/api/jury/start")
    async def start_jury(payload: JuryStartRequest | None = None) -> dict[str, Any]:
        request_started = time.monotonic()
        session = require_session()
        async with jury_lock(session):
            return await start_jury_locked(session, payload, request_started)

    async def start_jury_locked(
        session: PracticeSession,
        payload: JuryStartRequest | None,
        request_started: float,
    ) -> dict[str, Any]:
        if session.exposition_status != EXPOSITION_FINISHED:
            raise HTTPException(
                status_code=409, detail="Primero termina la exposición."
            )
        words = session.exposition_words()
        if words < min_exposition_words:
            raise HTTPException(
                status_code=409,
                detail=(
                    "La exposición es demasiado breve para el jurado: "
                    f"se escucharon {words} palabras y se necesitan al menos "
                    f"{min_exposition_words}. Expón tus láminas antes de pasar "
                    "al jurado."
                ),
            )
        questions = (
            payload.questions_per_judge
            if payload and payload.questions_per_judge is not None
            else session.jury.questions_per_judge
        )
        if not 1 <= questions <= MAX_QUESTIONS_PER_JUDGE:
            raise HTTPException(
                status_code=422,
                detail="Las preguntas por juez deben ser 1 o 2.",
            )
        chapters = resolve_chapters(payload.chapters if payload else None)
        idle = session.jury.status == JURY_IDLE
        with store.lock:
            session.jury.questions_per_judge = questions
            if idle:
                session.jury_mode = JURY_MODE_EXPOSITION
        if idle:
            apply_chapter_scope(session, chapters)

        if session.jury.status == JURY_DONE:
            return {
                "jury": jury_payload(session),
                "result": {
                    "type": "done",
                    "done": True,
                    "verdict": None,
                    "final": session.jury.final,
                },
            }
        if session.jury.status == JURY_DELIBERATING:
            return {
                "jury": jury_payload(session),
                "result": {
                    "type": "deliberating",
                    "done": False,
                    "verdict": None,
                    "final": None,
                },
            }
        if session.jury.status != JURY_IDLE:
            return {
                "jury": jury_payload(session),
                "result": {
                    "type": "question",
                    "done": False,
                    "verdict": None,
                    "final": None,
                },
            }

        try:
            await run_in_threadpool(ensure_judge_sessions, session)
        except Exception as exc:
            logger.exception("Could not create judge brain sessions")
            raise HTTPException(
                status_code=502,
                detail=f"No se pudo iniciar el jurado: {exc}",
            ) from exc

        planned = session.jury.planned_first
        session.jury.planned_first = None
        if (
            planned is not None
            and session.jury.judges[planned].asked == 0
            and remaining_for(session.jury, session.jury.judges[planned]) > 0
        ):
            first_index = planned
        else:
            first_index = pick_next_main_index(session.jury)
        if first_index is None:
            raise HTTPException(
                status_code=409, detail="No quedan preguntas para el jurado."
            )
        asked = await ask_main_question(
            session, first_index, request_started=request_started
        )
        if asked is None:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Ningún juez respondió a tiempo. Intenta iniciar el "
                    "jurado de nuevo."
                ),
            )
        finalize_timings(session, request_started)
        return {
            "jury": jury_payload(session),
            "result": {
                "type": "question",
                "done": False,
                "verdict": None,
                "final": None,
                "notice": take_notice(session),
            },
        }

    @app.post("/api/jury/direct")
    async def start_direct_jury(
        payload: JuryStartRequest | None = None,
    ) -> dict[str, Any]:
        """Start the jury without an exposition: direct questions only."""
        request_started = time.monotonic()
        session = require_session()
        async with jury_lock(session):
            return await start_direct_jury_locked(
                session, payload, request_started
            )

    async def start_direct_jury_locked(
        session: PracticeSession,
        payload: JuryStartRequest | None,
        request_started: float,
    ) -> dict[str, Any]:
        questions = (
            payload.questions_per_judge
            if payload and payload.questions_per_judge is not None
            else session.jury.questions_per_judge
        )
        if not 1 <= questions <= MAX_QUESTIONS_PER_JUDGE:
            raise HTTPException(
                status_code=422,
                detail="Las preguntas por juez deben ser 1 o 2.",
            )
        chapters = resolve_chapters(payload.chapters if payload else None)
        if session.jury.status == JURY_DONE:
            return {
                "jury": jury_payload(session),
                "result": {
                    "type": "done",
                    "done": True,
                    "verdict": None,
                    "final": session.jury.final,
                },
            }
        if session.jury.status == JURY_DELIBERATING:
            return {
                "jury": jury_payload(session),
                "result": {
                    "type": "deliberating",
                    "done": False,
                    "verdict": None,
                    "final": None,
                },
            }
        if session.jury.status != JURY_IDLE:
            return {
                "jury": jury_payload(session),
                "result": {
                    "type": "question",
                    "done": False,
                    "verdict": None,
                    "final": None,
                },
            }

        with store.lock:
            session.jury.questions_per_judge = questions
            session.jury_mode = JURY_MODE_DIRECT
            session.exposition_status = EXPOSITION_FINISHED
            if session.exposition_finished_at is None:
                session.exposition_finished_at = time.time()
        apply_chapter_scope(session, chapters)

        try:
            await run_in_threadpool(ensure_judge_sessions, session)
        except Exception as exc:
            logger.exception("Could not create judge brain sessions")
            raise HTTPException(
                status_code=502,
                detail=f"No se pudo iniciar el jurado: {exc}",
            ) from exc
        # No per-judge warm-up here: it blocked the request for ~25 s. The shared
        # prompt cache is warmed once at app startup (main), and the prefetched
        # first questions below warm each judge session. Mark warm-up as done so
        # no concurrent warm-up ask can cross replies in the same session.
        with warmup_lock:
            warmup_started.add(session.id)

        greeting_text = render_direct_greeting(
            direct_opening_line, student_name
        )
        greeting_audio = await run_in_threadpool(
            speak, greeting_text, "juez_presidente"
        )
        with store.lock:
            session.jury.greeting = {
                "text": greeting_text,
                "audio_url": (
                    f"/api/audio/{greeting_audio}" if greeting_audio else None
                ),
            }

        first_index = pick_next_main_index(session.jury)
        if first_index is None:
            raise HTTPException(
                status_code=409, detail="No quedan preguntas para el jurado."
            )
        await run_in_threadpool(
            schedule_first_question_prefetches, session, first_index
        )
        asked = await ask_main_question(
            session, first_index, request_started=request_started
        )
        if asked is None:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Ningún juez respondió a tiempo. Intenta iniciar el "
                    "jurado de nuevo."
                ),
            )
        finalize_timings(session, request_started)
        return {
            "jury": jury_payload(session),
            "result": {
                "type": "question",
                "done": False,
                "verdict": None,
                "final": None,
                "notice": take_notice(session),
            },
        }

    def ensure_answerable_turn(
        session: PracticeSession, turn_id: int | None
    ) -> None:
        """Fail fast on a stale turn before spending time on transcription."""
        jury = session.jury
        if jury.status != JURY_AWAITING_ANSWER or not jury.current_question:
            return
        if jury.current_turn_id is not None and turn_id != jury.current_turn_id:
            raise HTTPException(
                status_code=409, detail=DUPLICATE_ANSWER_DETAIL
            )

    async def handle_answer(
        session: PracticeSession,
        text: str,
        stt_ms: float = 0.0,
        turn_id: int | None = None,
        request_started: float | None = None,
    ) -> dict[str, Any]:
        if request_started is None:
            request_started = time.monotonic()
        text = (text or "").strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail="No se entendió tu respuesta. Repite, por favor.",
            )
        jury = session.jury
        if jury.status != JURY_AWAITING_ANSWER or not jury.current_question:
            raise HTTPException(
                status_code=409, detail="El jurado no está esperando una respuesta."
            )
        # T11 idempotency: the answer must name the question it answers, and an
        # identical retry for that same turn never records a second exchange.
        current_turn = jury.current_turn_id
        if current_turn is not None and turn_id != current_turn:
            raise HTTPException(
                status_code=409, detail=DUPLICATE_ANSWER_DETAIL
            )
        if (
            current_turn is not None
            and jury.last_answer_turn_id == current_turn
            and jury.last_answer_text == normalize_answer(text)
        ):
            raise HTTPException(
                status_code=409, detail=DUPLICATE_ANSWER_DETAIL
            )
        judge_index = jury.current_judge
        judge = jury.judges[judge_index]
        question = jury.current_question
        kind = jury.turn_kind or JURY_TURN_QUESTION

        with store.lock:
            next_order = sum(len(item.turns) for item in jury.judges) + 1
            judge.turns.append(
                JuryTurn(
                    question=question,
                    answer=text,
                    follow_up=(kind == JURY_TURN_FOLLOW_UP),
                    kind=kind,
                    order=next_order,
                )
            )
            jury.last_answer_turn_id = current_turn
            jury.last_answer_text = normalize_answer(text)

        result: dict[str, Any] = {
            "type": "question",
            "done": False,
            "verdict": None,
            "final": None,
        }

        def finish() -> dict[str, Any]:
            finalize_timings(session, request_started)
            result["notice"] = take_notice(session)
            return {"jury": jury_payload(session), "result": result}

        # The original asker keeps its single follow-up, as before.
        if kind == JURY_TURN_QUESTION and should_follow_up(jury, judge):
            if await ask_follow_up(
                session,
                judge_index,
                text,
                stt_ms=stt_ms,
                request_started=request_started,
            ):
                return finish()

        # After an answer another judge may interject; one per main question.
        if kind != JURY_TURN_INTERJECTION and picker.random() < interjection_rate:
            interjector_index = pick_interjection_index(jury, judge_index)
            if await ask_interjection(
                session,
                interjector_index,
                judge.name,
                question,
                text,
                stt_ms=stt_ms,
                request_started=request_started,
            ):
                return finish()

        next_index = pick_next_main_index(jury, avoid=jury.last_main_judge)
        if next_index is not None:
            asked = await ask_main_question(
                session,
                next_index,
                stt_ms=stt_ms,
                request_started=request_started,
            )
            if asked is not None:
                return finish()

        # Budgets exhausted (or every judge stalled): the jury deliberates over
        # the whole defense. The UI polls the session and shows the banner.
        for known_judge in jury.judges:
            await settle_prefetch(session, known_judge)
        clear_prefetches(session.id)
        with store.lock:
            jury.status = JURY_DELIBERATING
            jury.current_question = None
            jury.current_audio = None
            jury.current_audio_chunks = []
            jury.current_role = None
            jury.turn_kind = None
            jury.current_turn_id = None
            jury.last_timings = None
        deliberation_background.submit(collect_deliberation, session)
        result["type"] = "deliberating"
        result["done"] = False
        result["notice"] = take_notice(session)
        return {"jury": jury_payload(session), "result": result}

    @app.post("/api/jury/answer")
    async def answer_with_text(payload: TextAnswerRequest) -> dict[str, Any]:
        request_started = time.monotonic()
        session = require_session()
        async with jury_lock(session):
            return await handle_answer(
                session,
                payload.text,
                turn_id=payload.turn_id,
                request_started=request_started,
            )

    @app.post("/api/jury/answer/audio")
    async def answer_with_audio(
        file: UploadFile = File(...),
        turn_id: int | None = Form(None),
    ) -> dict[str, Any]:
        request_started = time.monotonic()
        session = require_session()
        async with jury_lock(session):
            ensure_answerable_turn(session, turn_id)
            stt_started = time.monotonic()
            text = await transcribe_upload(file, "answer")
            stt_ms = (time.monotonic() - stt_started) * 1000.0
            return await handle_answer(
                session,
                text,
                stt_ms=stt_ms,
                turn_id=turn_id,
                request_started=request_started,
            )

    @app.get("/api/jury/verdict/markdown")
    async def get_jury_verdict_markdown():
        """Download the final verdict as Markdown (S15)."""
        session = require_session()
        text = jury_verdict_markdown(session, heading="#")
        if not text:
            raise HTTPException(
                status_code=409,
                detail="El jurado todavía no ha deliberado.",
            )
        return PlainTextResponse(
            text + "\n",
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="veredicto-{session.id}.md"'
                )
            },
        )

    # -- evaluator feedback ------------------------------------------------

    async def evaluate_slide(session: PracticeSession, slide: SlideState) -> dict[str, Any]:
        transcript = (slide.transcript or "").strip()
        image = slides_path / slide.image

        def call_brain() -> str:
            files = [image_attachment(image)] if image.is_file() else []
            evaluator_session = brain.create_session("evaluador")
            return brain.ask(
                evaluator_session,
                evaluator_prompt(
                    slide.number,
                    slide.text,
                    transcript,
                    has_image=bool(files),
                    previous=slide.attempts[-1] if slide.attempts else None,
                ),
                files=files,
            )

        try:
            raw = await run_in_threadpool(call_brain)
        except Exception as exc:
            logger.exception("Evaluator failed for slide %s", slide.number)
            raise HTTPException(
                status_code=502,
                detail=f"El evaluador no respondió: {exc}",
            ) from exc
        feedback = parse_evaluator_reply(raw, slide.number)
        feedback["muletillas"] = count_muletillas(slide.transcript)
        feedback["muletillas_detalle"] = muletillas_breakdown(slide.transcript)
        feedback["palabras_por_minuto"] = words_per_minute(
            slide.transcript, slide.duration_seconds
        )
        with store.lock:
            slide.feedback = feedback
        return feedback

    @app.post("/api/feedback/global")
    async def evaluate_global() -> dict[str, Any]:
        session = require_session()
        if session.exposition_status != EXPOSITION_FINISHED:
            raise HTTPException(
                status_code=409, detail="Primero termina la exposición."
            )
        if session.exposition_words() == 0:
            raise HTTPException(
                status_code=409,
                detail="Sin exposición suficiente para el informe global.",
            )

        def call_brain() -> str:
            evaluator_session = brain.create_session("evaluador")
            return brain.ask(
                evaluator_session,
                global_feedback_prompt(
                    session, target_minutes_min, target_minutes_max
                ),
            )

        try:
            raw = await run_in_threadpool(call_brain)
        except Exception as exc:
            logger.exception("Global evaluator failed")
            raise HTTPException(
                status_code=502,
                detail=f"El evaluador no respondió: {exc}",
            ) from exc
        parsed = parse_global_reply(raw or "")
        report = build_global_report(
            session, parsed, target_minutes_min, target_minutes_max
        )
        with store.lock:
            session.global_report = report
        store.save(session)
        return {"global": report, "report": feedback_report(session)}

    @app.post("/api/feedback/{number}")
    async def evaluate_one(
        number: int, payload: SlideFeedbackRequest | None = None
    ) -> dict[str, Any]:
        session = require_session()
        slide = find_slide(session, number)
        if slide is None:
            raise HTTPException(status_code=404, detail="Lámina no encontrada.")
        if count_exposition_words(slide.transcript) == 0:
            raise HTTPException(
                status_code=409,
                detail="Sin exposición suficiente en esta lámina.",
            )
        feedback = await evaluate_slide(session, slide)
        store.save(session)
        # S21: the tutor can say the feedback out loud right after the take.
        spoken_text = None
        audio = None
        if payload is not None and payload.speak:
            spoken_text = tutor_feedback_speech(feedback)
            audio = await run_in_threadpool(speak, spoken_text, "tutor")
        return {
            "slide": slide.number,
            "feedback": feedback,
            "spoken_text": spoken_text,
            "audio_url": f"/api/audio/{audio}" if audio else None,
            "report": feedback_report(session),
        }

    @app.post("/api/feedback")
    async def evaluate_all() -> dict[str, Any]:
        session = require_session()
        pending = [
            slide
            for slide in session.slides
            if count_exposition_words(slide.transcript) > 0 and slide.feedback is None
        ]
        if not pending:
            raise HTTPException(
                status_code=409,
                detail=(
                    "No hay láminas con exposición suficiente pendientes de evaluar."
                ),
            )
        for slide in pending:
            await evaluate_slide(session, slide)
        store.save(session)
        return {"report": feedback_report(session)}

    @app.get("/api/progress")
    async def get_progress() -> dict[str, Any]:
        # Learning loop: how the saved rehearsals compare over time.
        return await run_in_threadpool(summarize_practices, sessions_path)

    @app.get("/api/feedback")
    async def get_feedback() -> dict[str, Any]:
        return {"report": feedback_report(require_session())}

    @app.get("/api/feedback/markdown")
    async def get_feedback_markdown():
        session = require_session()
        return PlainTextResponse(
            feedback_markdown(session),
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="retroalimentacion-{session.id}.md"'
                )
            },
        )

    # -- tutor -------------------------------------------------------------

    async def tutor_reply(
        session: PracticeSession,
        text: str,
        speak_reply: bool,
        slide_number: int | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        text = (text or "").strip()
        raw_images = list(images or [])
        if not text and not raw_images and slide_number is None:
            raise HTTPException(status_code=422, detail="Escribe tu pregunta.")
        if len(raw_images) > MAX_TUTOR_IMAGES:
            raise HTTPException(
                status_code=422,
                detail=f"Puedes enviar hasta {MAX_TUTOR_IMAGES} imágenes por mensaje.",
            )
        decoded: list[tuple[str, bytes]] = []
        for value in raw_images:
            try:
                decoded.append(decode_image_data_url(value))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        shown_slide = None
        files: list[dict[str, str]] = []
        if slide_number is not None:
            shown_slide = find_slide(session, slide_number)
            if shown_slide is None:
                raise HTTPException(status_code=404, detail="Lámina no encontrada.")
            slide_image = slides_path / shown_slide.image
            if slide_image.is_file():
                files.append(image_attachment(slide_image))
        for index, (mime, data) in enumerate(decoded, start=1):
            encoded = base64.b64encode(data).decode("ascii")
            files.append(
                {
                    "uri": f"data:{mime};base64,{encoded}",
                    "name": f"imagen-{index}{IMAGE_MIME_SUFFIXES[mime]}",
                }
            )
        prompt = tutor_prompt(text, shown_slide, len(decoded))
        if not session.tutor_session_id:
            try:
                session_id = await run_in_threadpool(brain.create_session, "tutor")
            except Exception as exc:
                logger.exception("Could not create tutor brain session")
                raise HTTPException(
                    status_code=502,
                    detail=f"No se pudo iniciar el tutor: {exc}",
                ) from exc
            with store.lock:
                session.tutor_session_id = session_id
        try:
            reply = await run_in_threadpool(
                lambda: brain.ask(session.tutor_session_id, prompt, files=files)
            )
        except Exception as exc:
            logger.exception("Tutor failed")
            raise HTTPException(
                status_code=502, detail=f"El tutor no respondió: {exc}"
            ) from exc
        reply = (reply or "").strip()
        if not reply:
            raise HTTPException(
                status_code=502, detail="El tutor devolvió una respuesta vacía."
            )
        audio = (
            await run_in_threadpool(speak, reply, "tutor") if speak_reply else None
        )
        image_urls: list[str] = []
        if decoded:
            tutor_images_path.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000)
            for index, (mime, data) in enumerate(decoded, start=1):
                name = f"tutor-{session.id}-{stamp}-{index}{IMAGE_MIME_SUFFIXES[mime]}"
                (tutor_images_path / name).write_bytes(data)
                image_urls.append(f"/api/tutor/images/{name}")
        with store.lock:
            session.tutor_messages.append(
                TutorMessage(
                    role="user", text=text, images=image_urls, slide=slide_number
                )
            )
            session.tutor_messages.append(TutorMessage(role="tutor", text=reply))
        return {
            "reply": reply,
            "audio_url": f"/api/audio/{audio}" if audio else None,
            "messages": [asdict(message) for message in session.tutor_messages],
        }

    @app.get("/api/tutor")
    async def get_tutor() -> dict[str, Any]:
        session = require_session()
        return {"messages": [asdict(message) for message in session.tutor_messages]}

    @app.post("/api/tutor/message")
    async def tutor_message(payload: TutorMessageRequest) -> dict[str, Any]:
        return await tutor_reply(
            require_session(),
            payload.text,
            payload.speak,
            slide_number=payload.slide,
            images=payload.images,
        )

    @app.post("/api/tutor/audio")
    async def tutor_audio(
        file: UploadFile = File(...),
        speak: bool = Form(False),
        slide: int | None = Form(None),
        images: str = Form("[]"),
    ) -> dict[str, Any]:
        session = require_session()
        try:
            image_list = json.loads(images or "[]")
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422, detail="Las imágenes no llegaron bien."
            ) from exc
        if not isinstance(image_list, list) or not all(
            isinstance(item, str) for item in image_list
        ):
            raise HTTPException(
                status_code=422, detail="Las imágenes no llegaron bien."
            )
        text = await transcribe_upload(file, "tutor")
        return await tutor_reply(
            session, text, speak, slide_number=slide, images=image_list
        )

    @app.get("/api/tutor/images/{filename}")
    async def get_tutor_image(filename: str):
        if Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Imagen no encontrada.")
        path = tutor_images_path / filename
        mime = SUFFIX_IMAGE_MIMES.get(path.suffix.lower())
        if mime is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Imagen no encontrada.")
        return FileResponse(path, media_type=mime)

    # -- generated audio ---------------------------------------------------

    @app.get("/api/jury/fillers")
    async def jury_fillers() -> dict[str, list[str]]:
        """Short reaction clips per judge, pre-synthesized with its own voice."""
        return await run_in_threadpool(prepare_fillers_once)

    @app.get("/api/audio/fillers/{filename}")
    async def get_filler_audio(filename: str):
        if Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Audio no encontrado.")
        path = fillers_path / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Audio no encontrado.")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/audio/{filename}")
    async def get_audio(filename: str):
        if Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Audio no encontrado.")
        path = audio_path / filename
        # S19: a streamed chunk URL can be requested before its synthesis ends.
        # Wait (bounded) for the pending file instead of 404ing the first chunk.
        if not (path.is_file() and path.stat().st_size > 0):
            with pending_audio_lock:
                future = pending_audio.get(filename)
            if future is not None:
                try:
                    await run_in_threadpool(future.result, AUDIO_WAIT_SECONDS)
                except Exception:
                    pass
        if not (path.is_file() and path.stat().st_size > 0):
            raise HTTPException(status_code=404, detail="Audio no encontrado.")
        return FileResponse(path, media_type="audio/wav")

    # -- static UI ---------------------------------------------------------

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        index_path = WEB_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=404, detail="Falta web/index.html en este checkout."
            )
        return FileResponse(index_path)

    return app


def main() -> None:
    """Run the app locally with the real OpenCode brain, whisper STT, and say TTS."""
    import uvicorn

    from app import session as session_module
    from app import stt as stt_module
    from app import tts as tts_module
    from app.brain import REPO_ROOT, Brain, load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config()
    tts_module.configure_voices(config["roles"])
    session_module.configure_judges(config["roles"])
    slides_dir = REPO_ROOT / config["slides"]["dir"]
    if not any(slides_dir.glob("slide-*.txt")):
        raise SystemExit("Run `.venv/bin/python scripts/build_context.py` first.")

    min_exposition_words = int(
        (config.get("jury") or {}).get("min_exposition_words", 80)
    )
    jury_config = config.get("jury") or {}
    student_config = config.get("student") or {}
    student_name = str(
        student_config.get("name") or DEFAULT_STUDENT_NAME
    ).strip()
    opening_line = str(jury_config.get("opening_line") or DEFAULT_OPENING_LINE)
    target_minutes_min = int(
        jury_config.get("target_minutes_min", DEFAULT_TARGET_MINUTES_MIN)
    )
    target_minutes_max = int(
        jury_config.get("target_minutes_max", DEFAULT_TARGET_MINUTES_MAX)
    )
    interjection_probability = float(
        jury_config.get(
            "interjection_probability", DEFAULT_INTERJECTION_PROBABILITY
        )
    )
    grading = jury_config.get("grading") or {}
    deadlines = jury_config.get("deadlines") or {}
    direct_opening_line = str(
        jury_config.get("direct_opening_line") or DEFAULT_DIRECT_OPENING_LINE
    )
    tutor_name = str(
        ((config.get("roles") or {}).get("tutor") or {}).get("name")
        or DEFAULT_TUTOR_NAME
    )
    brain = Brain(config).start()

    chapters = load_chapter_catalog(
        REPO_ROOT / config["context"]["dir"] / DEFAULT_CHAPTERS_FILENAME
    )

    def warm_shared_cache() -> None:
        # Every role shares the same context prefix, so one throwaway ask warms
        # the provider cache for all judges, the evaluator and the tutor.
        try:
            brain.ask(brain.create_session("tutor"), "Responde solo: listo")
            logger.info("Shared prompt cache warmed")
        except Exception:
            logger.warning("Startup cache warm-up failed", exc_info=True)

    def warm_xtts_engine() -> None:
        # Loading the XTTS model costs seconds; do it before the first turn.
        try:
            tts_module.warm_up_xtts()
            logger.info("XTTS warm-up ready")
        except Exception:
            logger.warning("XTTS warm-up failed", exc_info=True)

    threading.Thread(target=warm_shared_cache, daemon=True).start()
    threading.Thread(target=warm_xtts_engine, daemon=True).start()
    try:
        app = create_app(
            brain=brain,
            stt=stt_module,
            tts=tts_module,
            slides_dir=slides_dir,
            audio_dir=REPO_ROOT / "data" / "audio",
            session_dir=REPO_ROOT / "data" / "practicas",
            min_exposition_words=min_exposition_words,
            student_name=student_name,
            opening_line=opening_line,
            target_minutes_min=target_minutes_min,
            target_minutes_max=target_minutes_max,
            interjection_probability=interjection_probability,
            tutor_name=tutor_name,
            prepare_fillers_on_startup=True,
            grading=grading,
            direct_opening_line=direct_opening_line,
            deadlines=deadlines,
            chapters=chapters,
        )
        # uvicorn re-raises SIGINT/SIGTERM after shutdown, so the `finally`
        # below may never run; stop the OpenCode child during app shutdown.
        app.router.on_shutdown.append(brain.stop)
        port = int(os.environ.get("JURADO_PORT", "8000"))
        print(f"Simulador de jurado listo: http://127.0.0.1:{port}")
        uvicorn.run(app, host="127.0.0.1", port=port)
    finally:
        brain.stop()


if __name__ == "__main__":
    main()
