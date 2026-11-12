"""Practice-session state and JSON persistence.

The simulator keeps one practice session in memory (single local user) and
writes it to ``<session_dir>/<timestamp>-<id>.json`` once the practice ends.
Every value in the saved payload is JSON-serializable so the file can be
re-read or inspected without the app.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JUDGE_ROLES: tuple[str, ...] = ("juez_metodologo", "juez_tecnico", "juez_presidente")

# UI metadata for each judge. The behavior/persona prompts live in the brain
# configuration (app/brain.py); these strings are what the page displays.
JUDGES: tuple[dict[str, str], ...] = (
    {
        "role": "juez_metodologo",
        "name": "Dra. Elena Ríos",
        "persona": (
            "Metodóloga exigente: cuestiona el diseño de la investigación, la "
            "validez de los instrumentos, la muestra y la coherencia entre "
            "objetivos, metodología y conclusiones."
        ),
    },
    {
        "role": "juez_tecnico",
        "name": "Ing. Marco Salas",
        "persona": (
            "Ingeniero técnico: pregunta por la implementación, la arquitectura, "
            "las tecnologías elegidas, el rendimiento y los detalles prácticos "
            "del sistema."
        ),
    },
    {
        "role": "juez_presidente",
        "name": "Dr. Rafael Ponce",
        "persona": (
            "Presidente del jurado: evalúa la claridad de la exposición, el "
            "dominio del tema, la justificación del aporte y el cierre de la "
            "defensa."
        ),
    },
)

TRANSCRIPTION_PENDING = "pending"
TRANSCRIPTION_RUNNING = "transcribing"
TRANSCRIPTION_DONE = "done"
TRANSCRIPTION_EMPTY = "empty"
TRANSCRIPTION_ERROR = "error"

EXPOSITION_RUNNING = "running"
EXPOSITION_FINISHED = "finished"

EXPOSITION_MODE_PRACTICE = "practice"
EXPOSITION_MODE_REAL = "real"

JURY_IDLE = "idle"
JURY_AWAITING_ANSWER = "awaiting_answer"
JURY_DELIBERATING = "deliberating"
JURY_DONE = "done"

JURY_MODE_EXPOSITION = "exposicion"
JURY_MODE_DIRECT = "directo"

# S15: the Venezuelan 1-20 scale and a pass mark of 10 are an assumption the
# user can change in config.json (jury.grading).
DEFAULT_GRADING: dict[str, Any] = {
    "scale_max": 20,
    "pass_mark": 10,
    "weights": {"exposicion": 0.5, "respuestas": 0.5},
}

INSUFFICIENT_EXPOSITION = "Sin exposición suficiente en esta lámina."

# Spoken fillers that do not count as exposition content.
FILLER_WORDS = frozenset(
    {
        "ok",
        "okay",
        "okey",
        "okei",
        "eh",
        "ehh",
        "ehhh",
        "este",
        "mm",
        "mmm",
        "mmmm",
        "em",
        "ah",
        "oh",
        "uh",
        "hmm",
    }
)

_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def count_exposition_words(text: str | None) -> int:
    """Count words made of letters, ignoring filler interjections."""
    if not text:
        return 0
    return sum(
        1
        for match in _WORD_PATTERN.finditer(text)
        if match.group(0).casefold() not in FILLER_WORDS
    )


# Spoken fillers that are real words: counted as muletillas, but they still
# count as exposition words so the minimum-exposition guard is unchanged.
FILLER_EXTRA_WORDS = frozenset({"digamos", "básicamente", "basicamente", "ajá", "aja"})
FILLER_PHRASES = frozenset({("o", "sea"), ("como", "que"), ("en", "plan")})


def muletillas_breakdown(text: str | None) -> dict[str, int]:
    """Count each spoken filler, most frequent first.

    Two-word fillers such as «o sea» are matched before single words, so the
    feedback can name the student's habit instead of just a total.
    """
    if not text:
        return {}
    words = [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]
    counts: dict[str, int] = {}
    index = 0
    while index < len(words):
        pair = tuple(words[index : index + 2])
        if pair in FILLER_PHRASES:
            key = " ".join(pair)
            index += 2
        elif words[index] in FILLER_WORDS or words[index] in FILLER_EXTRA_WORDS:
            key = words[index]
            index += 1
        else:
            index += 1
            continue
        counts[key] = counts.get(key, 0) + 1
    order = list(counts)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], order.index(item[0]))))


def count_muletillas(text: str | None) -> int:
    """Count spoken fillers (muletillas) in a transcript.

    Computed in Python, never by the evaluator model, so the number is
    reproducible and cannot be inflated by the LLM.
    """
    return sum(muletillas_breakdown(text).values())


def words_per_minute(
    text: str | None, duration_seconds: float | None
) -> int | None:
    """Content words per minute, or ``None`` when there is no duration."""
    if not duration_seconds or duration_seconds <= 0:
        return None
    words = count_exposition_words(text)
    return int(round(words / (float(duration_seconds) / 60.0)))


class NoSessionError(RuntimeError):
    """Raised when the app needs a practice session but none was started."""


@dataclass
class SlideState:
    """One slide of the deck plus the student's recorded material."""

    number: int
    image: str
    text: str = ""
    audio: str | None = None
    transcript: str | None = None
    duration_seconds: float | None = None
    transcription_status: str = TRANSCRIPTION_PENDING
    transcription_error: str | None = None
    feedback: dict[str, Any] | None = None
    # Earlier evaluated takes of this slide, oldest first (learning loop).
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def archive_feedback(self) -> None:
        """Move the current feedback into ``attempts`` before a new take."""
        if not self.feedback:
            return
        feedback = self.feedback
        self.attempts.append(
            {
                "puntaje": _as_score(feedback.get("puntaje")),
                "muletillas": feedback.get("muletillas"),
                "palabras_por_minuto": feedback.get("palabras_por_minuto"),
                "omisiones": _as_text_list(feedback.get("omisiones")),
                "correcciones": _as_text_list(feedback.get("correcciones")),
                "archived_at": time.time(),
            }
        )
        self.feedback = None


@dataclass
class JuryTurn:
    """One question/answer exchange with a judge."""

    question: str
    answer: str
    follow_up: bool = False
    kind: str = "pregunta"
    order: int = 0


@dataclass
class JudgeState:
    """Per-judge conversation state and final verdict."""

    role: str
    name: str
    persona: str
    brain_session_id: str | None = None
    asked: int = 0
    follow_ups: int = 0
    interjections: int = 0
    turns: list[JuryTurn] = field(default_factory=list)
    verdict: str | None = None
    score: float | None = None


def configure_judges(roles: dict[str, dict]) -> None:
    """Replace the displayed judge metadata with the personas in config.json."""
    global JUDGES
    judges = []
    for role in JUDGE_ROLES:
        settings = roles.get(role)
        if not settings:
            continue
        persona = f"{settings.get('role', '')}. {settings.get('tone', '')}".strip(". ")
        judges.append({"role": role, "name": settings.get("name", role), "persona": persona})
    if judges:
        JUDGES = tuple(judges)


def _default_judges() -> list[JudgeState]:
    return [JudgeState(**metadata) for metadata in JUDGES]


@dataclass
class JuryState:
    """Turn-taking state for the three-judge panel."""

    status: str = JURY_IDLE
    questions_per_judge: int = 2
    # S23: the judge whose opening question was prefetched asks first.
    planned_first: int | None = None
    current_judge: int = 0
    current_role: str | None = None
    turn_kind: str | None = None
    current_question: str | None = None
    current_audio: str | None = None
    # S19: the judge question is streamed as sentence chunks; the client plays
    # them back to back and `current_audio` is the first one (compatibility).
    current_audio_chunks: list[str] = field(default_factory=list)
    # Short user-facing notice when a turn had to be degraded (S19 deadlines).
    notice: str | None = None
    # T11: the id of the question currently awaiting an answer. The client
    # echoes it back so a duplicate submission cannot record a second turn.
    current_turn_id: int | None = None
    turn_seq: int = 0
    last_answer_turn_id: int | None = None
    last_answer_text: str | None = None
    last_main_judge: int | None = None
    last_timings: dict[str, Any] | None = None
    greeting: dict[str, Any] | None = None
    judges: list[JudgeState] = field(default_factory=_default_judges)
    final: dict[str, Any] | None = None

    def begin_turn(self) -> int:
        """Start a new answerable turn and return its stable id."""
        self.turn_seq += 1
        self.current_turn_id = self.turn_seq
        return self.current_turn_id


@dataclass
class TutorMessage:
    """One message of the tutor chat."""

    role: str
    text: str
    at: float = field(default_factory=time.time)
    # S22: URLs of the images the student attached and the slide shown.
    images: list[str] = field(default_factory=list)
    slide: int | None = None


@dataclass
class SlideMark:
    """A slide change inside one continuous recording.

    ``start_ms`` is milliseconds since the recording started.
    """

    slide: int
    start_ms: int


GLOBAL_RUBRIC: tuple[tuple[str, str], ...] = (
    ("dominio_del_tema", "Dominio del tema"),
    ("claridad_y_lenguaje", "Claridad y lenguaje"),
    ("adecuacion_al_tiempo", "Adecuación al tiempo"),
    ("coherencia_con_la_tesis", "Coherencia con la tesis"),
    ("uso_de_la_presentacion", "Uso de la presentación"),
)


@dataclass
class PracticeSession:
    """Complete state of one practice session."""

    id: str
    created_at: float
    slides: list[SlideState]
    exposition_status: str = EXPOSITION_RUNNING
    exposition_mode: str = EXPOSITION_MODE_PRACTICE
    jury_mode: str = JURY_MODE_EXPOSITION
    # S20: chosen chapter ids; empty means the whole thesis.
    jury_chapters: list[str] = field(default_factory=list)
    exposition_started_at: float | None = None
    exposition_finished_at: float | None = None
    slide_timeline: list[SlideMark] = field(default_factory=list)
    full_audio: str | None = None
    full_duration_seconds: float | None = None
    global_report: dict[str, Any] | None = None
    jury: JuryState = field(default_factory=JuryState)
    tutor_session_id: str | None = None
    tutor_messages: list[TutorMessage] = field(default_factory=list)
    saved_path: str | None = None
    save_name: str = ""

    def elapsed_seconds(self) -> float:
        if self.exposition_started_at is None:
            return 0.0
        end = self.exposition_finished_at or time.time()
        return max(0.0, end - self.exposition_started_at)

    def total_duration(self) -> float:
        return round(
            sum(slide.duration_seconds or 0.0 for slide in self.slides), 2
        )

    def exposition_transcript(self) -> str:
        parts = [
            f"Lámina {slide.number}:\n{slide.transcript.strip()}"
            for slide in self.slides
            if (slide.transcript or "").strip()
        ]
        return "\n\n".join(parts)

    def exposition_words(self) -> int:
        """Total content words across every slide transcript."""
        return sum(count_exposition_words(slide.transcript) for slide in self.slides)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_seconds"] = round(self.elapsed_seconds(), 2)
        payload["total_duration_seconds"] = self.total_duration()
        return payload


def apply_full_transcription(
    session: PracticeSession,
    timeline: list[SlideMark],
    segments: list[tuple[float, float, str]],
    total_ms: int | None = None,
) -> None:
    """Assign one continuous transcript to the slides in the timeline.

    Each whisper segment belongs to the slide whose interval contains the
    segment midpoint. Revisiting a slide appends to its transcript and adds its
    speaking time, so the per-slide feedback keeps working unchanged.
    """
    marks = sorted(
        (mark for mark in timeline if mark.start_ms >= 0),
        key=lambda mark: mark.start_ms,
    )
    if not marks:
        return

    cleaned: list[tuple[float, float, str]] = []
    for start, end, text in segments:
        content = (text or "").strip()
        if not content:
            continue
        start_s = max(0.0, float(start))
        end_s = max(start_s, float(end))
        cleaned.append((start_s, end_s, content))

    if total_ms is None or total_ms <= 0:
        total_ms = max(
            [marks[-1].start_ms]
            + [int(round(end * 1000)) for _, end, _ in cleaned]
        )

    durations: dict[int, float] = {}
    texts: dict[int, list[str]] = {}
    for index, mark in enumerate(marks):
        end = (
            marks[index + 1].start_ms
            if index + 1 < len(marks)
            else total_ms
        )
        end = min(end, max(total_ms, mark.start_ms))
        end = max(end, mark.start_ms)
        durations[mark.slide] = durations.get(mark.slide, 0.0) + (
            end - mark.start_ms
        ) / 1000.0
        texts.setdefault(mark.slide, [])

    for start_s, end_s, content in cleaned:
        midpoint_ms = (start_s + end_s) / 2.0 * 1000.0
        chosen = marks[0]
        for mark in marks:
            if mark.start_ms <= midpoint_ms:
                chosen = mark
            else:
                break
        texts[chosen.slide].append(content)

    for slide in session.slides:
        if slide.number not in durations:
            continue
        transcript = " ".join(texts.get(slide.number, [])).strip()
        slide.transcript = transcript or None
        slide.transcription_status = (
            TRANSCRIPTION_DONE if transcript else TRANSCRIPTION_EMPTY
        )
        slide.transcription_error = None
        slide.duration_seconds = round(durations[slide.number], 2)


class SessionStore:
    """Holds the current practice session and persists it as JSON."""

    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self.lock = threading.RLock()
        self._current: PracticeSession | None = None

    def start(
        self,
        slides: list[dict[str, Any]],
        questions_per_judge: int = 2,
    ) -> PracticeSession:
        with self.lock:
            now = time.time()
            session_id = uuid.uuid4().hex[:8]
            session = PracticeSession(
                id=session_id,
                created_at=now,
                slides=[
                    SlideState(
                        number=int(definition["number"]),
                        image=str(definition.get("image", "")),
                        text=str(definition.get("text", "")),
                    )
                    for definition in slides
                ],
                jury=JuryState(questions_per_judge=questions_per_judge),
            )
            session.save_name = (
                f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}"
                f"-{session_id}.json"
            )
            self._current = session
            return session

    @property
    def current(self) -> PracticeSession | None:
        with self.lock:
            return self._current

    def require(self) -> PracticeSession:
        session = self.current
        if session is None:
            raise NoSessionError("No hay una práctica en curso.")
        return session

    def save(self, session: PracticeSession | None = None) -> Path:
        with self.lock:
            session = session or self._current
            if session is None:
                raise NoSessionError("No hay una práctica en curso.")
            if not session.save_name:
                stamp = time.strftime(
                    "%Y%m%d-%H%M%S", time.localtime(session.created_at)
                )
                session.save_name = f"{stamp}-{session.id}.json"
            self.session_dir.mkdir(parents=True, exist_ok=True)
            path = self.session_dir / session.save_name
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
            session.saved_path = str(path)
            return path


def _breakdown_suffix(breakdown: Any) -> str:
    """Render ``{"o sea": 3}`` as ` (o sea ×3)` for the Markdown export."""
    if not isinstance(breakdown, dict) or not breakdown:
        return ""
    return " (" + ", ".join(f"{word} ×{count}" for word, count in breakdown.items()) + ")"


# Slides averaging under this score (0-10) are reported as weak across practices.
WEAK_SLIDE_THRESHOLD = 7.0


def _slide_scores(slide: dict[str, Any]) -> list[float]:
    """Every evaluated take of one saved slide: archived attempts plus the last."""
    scores = []
    for attempt in slide.get("attempts") or []:
        if isinstance(attempt, dict):
            score = _as_score(attempt.get("puntaje"))
            if score is not None:
                scores.append(score)
    feedback = slide.get("feedback")
    if isinstance(feedback, dict):
        score = _as_score(feedback.get("puntaje"))
        if score is not None:
            scores.append(score)
    return scores


def summarize_practices(directory: str | Path) -> dict[str, Any]:
    """Summarize saved practices so the student can see progress over time.

    Reads ``<directory>/*.json`` (unreadable files are skipped) and returns the
    measurable practices oldest first, the change between the last two jury
    grades, the slides that keep scoring low, and the filler habits overall.
    """
    practices: list[dict[str, Any]] = []
    slide_scores: dict[int, list[float]] = {}
    fillers: dict[str, int] = {}
    path = Path(directory)
    files = sorted(path.glob("*.json")) if path.is_dir() else []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        slides = [slide for slide in data.get("slides") or [] if isinstance(slide, dict)]
        final = (data.get("jury") or {}).get("final") if isinstance(data.get("jury"), dict) else None
        final = final if isinstance(final, dict) else {}
        jury_grade = _bounded_score(
            final.get("final_grade", final.get("average")),
            _bounded_score(final.get("scale_max"), 100.0) or 20.0,
        )
        report = data.get("global_report") if isinstance(data.get("global_report"), dict) else {}

        scores = []
        for slide in slides:
            taken = _slide_scores(slide)
            if taken:
                scores.append(taken[-1])
                try:
                    slide_scores.setdefault(int(slide.get("number")), []).extend(taken)
                except (TypeError, ValueError):
                    pass
        rubric = [
            score
            for score in (_as_score(item.get("puntaje")) for item in report.get("rubrica") or [] if isinstance(item, dict))
            if score is not None
        ]
        if jury_grade is None and not scores and not report:
            continue

        transcript = " ".join(str(slide.get("transcript") or "") for slide in slides)
        breakdown = muletillas_breakdown(transcript)
        for word, count in breakdown.items():
            fillers[word] = fillers.get(word, 0) + count
        practices.append(
            {
                "id": str(data.get("id") or file.stem),
                "created_at": data.get("created_at"),
                "mode": data.get("jury_mode"),
                "jury_grade": jury_grade,
                "scale_max": final.get("scale_max"),
                "mention": final.get("mention"),
                "slides_evaluated": len(scores),
                "slide_average": round(sum(scores) / len(scores), 1) if scores else None,
                "rubric_average": round(sum(rubric) / len(rubric), 1) if rubric else None,
                "words_per_minute": report.get("words_per_minute"),
                "muletillas": sum(breakdown.values()),
            }
        )

    practices.sort(key=lambda item: item["created_at"] or 0)
    grades = [item["jury_grade"] for item in practices if item["jury_grade"] is not None]
    weak = [
        {"number": number, "average": round(sum(values) / len(values), 1), "attempts": len(values)}
        for number, values in slide_scores.items()
        if sum(values) / len(values) < WEAK_SLIDE_THRESHOLD
    ]
    weak.sort(key=lambda item: (item["average"], -item["attempts"], item["number"]))
    return {
        "practices": practices,
        "jury_trend": round(grades[-1] - grades[-2], 1) if len(grades) >= 2 else None,
        "weak_slides": weak[:3],
        "muletillas": dict(sorted(fillers.items(), key=lambda item: -item[1])),
    }


def format_duration(seconds: float | None) -> str:
    """Format seconds as ``mm:ss`` for reports and the UI."""
    total = int(round(seconds or 0))
    return f"{total // 60:02d}:{total % 60:02d}"


def _as_score(value: Any) -> float | None:
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


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def build_global_report(
    session: PracticeSession,
    parsed: dict[str, Any],
    target_minutes_min: int = 15,
    target_minutes_max: int = 18,
) -> dict[str, Any]:
    """Merge the model's global verdict with Python-computed metrics.

    Duration, words per minute and muletillas are computed here so the LLM
    cannot invent them. Weak slides come from the model when it names real
    slides, otherwise from the lowest per-slide scores.
    """
    duration = session.total_duration()
    min_seconds = target_minutes_min * 60
    max_seconds = target_minutes_max * 60
    words = session.exposition_words()

    rubric_data = parsed.get("rubrica")
    if not isinstance(rubric_data, dict):
        rubric_data = {}
    rubrica = []
    for key, label in GLOBAL_RUBRIC:
        item = rubric_data.get(key)
        if not isinstance(item, dict):
            item = {}
        rubrica.append(
            {
                "key": key,
                "label": label,
                "puntaje": _as_score(item.get("puntaje")),
                "comentario": str(item.get("comentario") or "").strip(),
            }
        )

    prioridades = _as_text_list(parsed.get("prioridades"))[:3]

    known = {slide.number for slide in session.slides}
    weakest: list[int] = []
    raw_weak = parsed.get("laminas_debiles")
    if not isinstance(raw_weak, (list, tuple)):
        raw_weak = []
    for value in raw_weak:
        if isinstance(value, dict):
            value = value.get("slide", value.get("number"))
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number in known and number not in weakest:
            weakest.append(number)
        if len(weakest) == 3:
            break
    if not weakest:
        scored = [
            (slide.number, _as_score(slide.feedback.get("puntaje")))
            for slide in session.slides
            if slide.feedback
        ]
        scored = [
            (number, score) for number, score in scored if score is not None
        ]
        scored.sort(key=lambda item: (item[1], item[0]))
        weakest = [number for number, _ in scored[:3]]

    by_number = {slide.number: slide for slide in session.slides}
    laminas_debiles = []
    for number in weakest:
        feedback = by_number[number].feedback or {}
        laminas_debiles.append(
            {"number": number, "puntaje": _as_score(feedback.get("puntaje"))}
        )

    return {
        "generated_at": time.time(),
        "duration_seconds": round(duration, 2),
        "duration_label": format_duration(duration),
        "target": {
            "min_minutes": target_minutes_min,
            "max_minutes": target_minutes_max,
            "min_label": format_duration(min_seconds),
            "max_label": format_duration(max_seconds),
        },
        "within_target": bool(
            duration > 0 and min_seconds <= duration <= max_seconds
        ),
        "words": words,
        "words_per_minute": words_per_minute(
            " ".join(
                slide.transcript or ""
                for slide in session.slides
                if slide.transcript
            ),
            duration,
        ),
        "muletillas": sum(
            count_muletillas(slide.transcript) for slide in session.slides
        ),
        "muletillas_detalle": muletillas_breakdown(
            " ".join(slide.transcript or "" for slide in session.slides)
        ),
        "rubrica": rubrica,
        "prioridades": prioridades,
        "laminas_debiles": laminas_debiles,
        "parse_error": bool(parsed.get("parse_error")),
        "raw": parsed.get("raw"),
    }


def feedback_report(session: PracticeSession) -> dict[str, Any]:
    """Build the per-slide feedback report consumed by the UI and export."""
    slides: list[dict[str, Any]] = []
    evaluated = 0
    for slide in session.slides:
        feedback = slide.feedback or {}
        has_text = bool((slide.transcript or "").strip())
        has_content = count_exposition_words(slide.transcript) > 0
        if slide.feedback is not None:
            evaluated += 1
            status = "ok"
        elif has_content:
            status = "missing"
        elif has_text:
            status = "insufficient"
        else:
            status = "no_transcript"
        slides.append(
            {
                "number": slide.number,
                "duration_seconds": slide.duration_seconds,
                "duration_label": format_duration(slide.duration_seconds),
                "transcript": slide.transcript,
                "status": status,
                "message": (
                    INSUFFICIENT_EXPOSITION
                    if status in {"insufficient", "no_transcript"}
                    else ""
                ),
                "dijo": str(feedback.get("dijo") or "").strip(),
                "fortalezas": list(feedback.get("fortalezas") or []),
                "correcciones": list(feedback.get("correcciones") or []),
                "omisiones": _as_text_list(feedback.get("omisiones")),
                "errores_vs_tesis": _as_text_list(
                    feedback.get("errores_vs_tesis")
                ),
                "frase_sugerida": feedback.get("frase_sugerida") or "",
                "mejoras_vs_anterior": str(
                    feedback.get("mejoras_vs_anterior") or ""
                ).strip(),
                "attempts": list(slide.attempts),
                "puntaje": _as_score(feedback.get("puntaje")),
                "muletillas": count_muletillas(slide.transcript),
                "muletillas_detalle": muletillas_breakdown(slide.transcript),
                "palabras_por_minuto": words_per_minute(
                    slide.transcript, slide.duration_seconds
                ),
                "parse_error": bool(feedback.get("parse_error")),
                "raw": feedback.get("raw"),
            }
        )
    return {
        "session_id": session.id,
        "created_at": session.created_at,
        "slides": slides,
        "global": session.global_report,
        "totals": {
            "slides": len(slides),
            "evaluated": evaluated,
            "duration_seconds": session.total_duration(),
            "duration_label": format_duration(session.total_duration()),
        },
    }


def feedback_markdown(session: PracticeSession) -> str:
    """Render the feedback report as a downloadable Markdown document."""
    report = feedback_report(session)
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(session.created_at))
    lines = [
        "# Retroalimentación por lámina",
        "",
        f"Sesión: {session.id} · {stamp}",
        "",
        f"Duración total: {report['totals']['duration_label']}",
        "",
    ]
    if session.global_report:
        lines.extend(_global_markdown(session.global_report))
        lines.append("")
    for slide in report["slides"]:
        lines.append(
            f"## Lámina {slide['number']} · {slide['duration_label']}"
        )
        lines.append("")
        if slide["status"] in {"no_transcript", "insufficient"}:
            lines.append("_Sin exposición suficiente en esta lámina._")
        elif slide["status"] == "missing":
            lines.append("_Sin retroalimentación generada._")
        else:
            if slide["parse_error"]:
                lines.append(
                    "> No se pudo interpretar la respuesta del evaluador; "
                    "se muestra el texto original."
                )
                lines.append("")
            if slide["puntaje"] is not None:
                lines.append(f"**Puntaje:** {slide['puntaje']:g}/10")
                lines.append("")
            if slide["dijo"]:
                lines.append(f"**Dijo:** {slide['dijo']}")
                lines.append("")
            lines.append(
                f"**Muletillas:** {slide['muletillas']}"
                f"{_breakdown_suffix(slide.get('muletillas_detalle'))} · "
                f"**Palabras por minuto:** "
                f"{slide['palabras_por_minuto'] if slide['palabras_por_minuto'] is not None else '—'}"
            )
            lines.append("")
            lines.append("**Fortalezas**")
            if slide["fortalezas"]:
                lines.extend(f"- {item}" for item in slide["fortalezas"])
            else:
                lines.append("- Sin observaciones.")
            lines.append("")
            lines.append("**Correcciones**")
            if slide["correcciones"]:
                lines.extend(f"- {item}" for item in slide["correcciones"])
            else:
                lines.append("- Sin correcciones.")
            lines.append("")
            lines.append("**Omisiones**")
            if slide["omisiones"]:
                lines.extend(f"- {item}" for item in slide["omisiones"])
            else:
                lines.append("- Sin omisiones detectadas.")
            lines.append("")
            lines.append("**Errores vs. la tesis**")
            if slide["errores_vs_tesis"]:
                lines.extend(f"- {item}" for item in slide["errores_vs_tesis"])
            else:
                lines.append("- Sin contradicciones detectadas.")
            lines.append("")
            if slide["frase_sugerida"]:
                lines.append(f"**Frase sugerida:** {slide['frase_sugerida']}")
            if slide["parse_error"] and slide["raw"]:
                lines.append("")
                lines.append("Respuesta original del evaluador:")
                lines.append("")
                lines.append(f"```\n{slide['raw']}\n```")
        lines.append("")
    verdict = jury_verdict_markdown(session)
    if verdict:
        lines.extend(["", verdict])
    return "\n".join(lines).rstrip() + "\n"


def _global_markdown(report: dict[str, Any]) -> list[str]:
    """Render the global report section of the Markdown export."""
    target = report.get("target") or {}
    estado = (
        "dentro del objetivo"
        if report.get("within_target")
        else "fuera del objetivo"
    )
    rpm = report.get("words_per_minute")
    lines = [
        "## Informe global",
        "",
        f"- Tiempo total: {report.get('duration_label', '00:00')} "
        f"(objetivo {target.get('min_label', '15:00')}–"
        f"{target.get('max_label', '18:00')}): {estado}",
        f"- Palabras por minuto: {rpm if rpm is not None else '—'}",
        f"- Muletillas: {report.get('muletillas', 0)}"
        f"{_breakdown_suffix(report.get('muletillas_detalle'))}",
        "",
        "### Rúbrica",
        "",
    ]
    for item in report.get("rubrica") or []:
        score = item.get("puntaje")
        score_label = (
            f"{score:g}/10" if isinstance(score, (int, float)) else "sin puntaje"
        )
        comment = f" — {item['comentario']}" if item.get("comentario") else ""
        lines.append(f"- {item.get('label')}: {score_label}{comment}")
    lines.extend(["", "### Prioridades", ""])
    priorities = report.get("prioridades") or []
    if priorities:
        lines.extend(
            f"{index}. {item}" for index, item in enumerate(priorities, start=1)
        )
    else:
        lines.append("- Sin prioridades registradas.")
    lines.extend(["", "### Láminas más débiles", ""])
    weak = report.get("laminas_debiles") or []
    if weak:
        parts = []
        for item in weak:
            score = item.get("puntaje")
            suffix = f" ({score:g}/10)" if isinstance(score, (int, float)) else ""
            parts.append(f"Lámina {item.get('number')}{suffix}")
        lines.append("- " + ", ".join(parts))
    else:
        lines.append("- Sin láminas calificadas.")
    if report.get("parse_error"):
        lines.extend(
            [
                "",
                "> No se pudo interpretar la respuesta del evaluador; "
                "se muestran solo las métricas calculadas.",
            ]
        )
        if report.get("raw"):
            lines.extend(["", f"```\n{report['raw']}\n```"])
    return lines


# ---------------------------------------------------------------------------
# S15 final grade (computed in Python, never by the model)
# ---------------------------------------------------------------------------


def _bounded_score(value: Any, maximum: float) -> float | None:
    """Coerce a model score to a 0-``maximum`` float, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return round(min(float(maximum), max(0.0, score)), 1)


def grading_settings(
    grading: dict[str, Any] | None,
) -> tuple[float, float, float, float]:
    settings: dict[str, Any] = dict(DEFAULT_GRADING)
    if isinstance(grading, dict):
        settings.update(grading)
    try:
        scale = float(settings.get("scale_max"))
    except (TypeError, ValueError):
        scale = 20.0
    if scale <= 0:
        scale = 20.0
    try:
        pass_mark = float(settings.get("pass_mark", 10))
    except (TypeError, ValueError):
        pass_mark = 10.0
    pass_mark = min(scale, max(0.0, pass_mark))
    weights = settings.get("weights")
    if not isinstance(weights, dict):
        weights = {}
    try:
        weight_exposition = max(0.0, float(weights.get("exposicion", 0.5)))
    except (TypeError, ValueError):
        weight_exposition = 0.5
    try:
        weight_answers = max(0.0, float(weights.get("respuestas", 0.5)))
    except (TypeError, ValueError):
        weight_answers = 0.5
    if weight_exposition + weight_answers <= 0:
        weight_exposition = weight_answers = 0.5
    return scale, pass_mark, weight_exposition, weight_answers


def jury_mention(grade: float | None, pass_mark: float = 10.0) -> str:
    """Return the Spanish mention for a final grade on the configured scale."""
    if grade is None:
        return ""
    if grade >= 18:
        return "Excelente"
    if grade >= 15:
        return "Muy bueno"
    if grade >= 12:
        return "Bueno"
    if grade >= pass_mark:
        return "Suficiente"
    return "Insuficiente"


def compute_jury_grade(
    evaluations: list[dict[str, Any]],
    grading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute every judge's grade and the final grade from parsed evaluations.

    The model only provides the two component scores; the weighted nota, the
    mean, the pass/fail result and the mention are always computed here. A judge
    without any usable score is kept in the list with ``nota: None`` and left
    out of the mean, so one malformed reply cannot invent a grade. In direct
    mode (no exposition) the exposition score is ``None`` and the nota is the
    answer score alone.
    """
    scale, pass_mark, weight_exposition, weight_answers = grading_settings(
        grading
    )
    weight_sum = weight_exposition + weight_answers

    judges: list[dict[str, Any]] = []
    notas: list[float] = []
    for evaluation in evaluations or []:
        row = dict(evaluation) if isinstance(evaluation, dict) else {}
        exposition = _bounded_score(row.get("exposicion"), scale)
        answers = _bounded_score(row.get("respuestas"), scale)
        if exposition is None and answers is None:
            nota: float | None = None
        elif exposition is None:
            nota = round(answers, 1)
        elif answers is None:
            nota = round(exposition, 1)
        else:
            nota = round(
                (weight_exposition * exposition + weight_answers * answers)
                / weight_sum,
                1,
            )
        row["exposicion"] = exposition
        row["respuestas"] = answers
        row["nota"] = nota
        if nota is not None:
            notas.append(nota)
        judges.append(row)

    final_grade = round(sum(notas) / len(notas), 1) if notas else None
    passed = final_grade is not None and final_grade >= pass_mark
    if final_grade is None:
        result = "Sin calificación"
    elif passed:
        result = "Aprobado"
    else:
        result = "Reprobado"
    no_exposition = bool(judges) and all(
        judge["exposicion"] is None for judge in judges
    )
    return {
        "judges": judges,
        "final_grade": final_grade,
        "average": final_grade,
        "result": result,
        "mention": jury_mention(final_grade, pass_mark),
        "passed": passed,
        "scale_max": scale,
        "pass_mark": pass_mark,
        "weights": {
            "exposicion": weight_exposition,
            "respuestas": weight_answers,
        },
        "no_exposition": no_exposition,
    }


_TURN_KIND_MARKDOWN = {
    "pregunta": "Pregunta",
    "repregunta": "Repregunta",
    "intervencion": "Intervención",
}


def _score_label(value: Any, scale: float) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}/{scale:g}"
    return "—"


def jury_verdict_markdown(
    session: PracticeSession, heading: str = "##"
) -> str:
    """Render the final jury verdict (S15) as Markdown.

    Returns an empty string when the jury has not finished deliberating.
    """
    final = session.jury.final
    if not final:
        return ""
    scale = final.get("scale_max") or 20
    sub = heading + "#"
    lines = [f"{heading} Veredicto del jurado", ""]
    grade = final.get("final_grade")
    lines.append(
        f"**Calificación final:** {_score_label(grade, scale)}"
        if grade is not None
        else "**Calificación final:** sin calificación"
    )
    result = final.get("result") or "Sin calificación"
    mention = final.get("mention")
    lines.append(f"**Resultado:** {result}" + (f" · {mention}" if mention else ""))
    if final.get("no_exposition"):
        lines.extend(
            [
                "",
                "_Sin exposición (modo directo): la nota se calcula solo con "
                "las respuestas._",
            ]
        )
    chapters = final.get("chapters") or []
    if chapters:
        lines.extend(["", f"**Preguntas sobre: {', '.join(chapters)}**"])
    if final.get("closing"):
        lines.extend(["", f"> {final['closing']}"])

    lines.extend(["", f"{sub} Evaluación por juez", ""])
    verdicts = final.get("verdicts") or []
    if not verdicts:
        lines.append("- Sin evaluaciones registradas.")
    for item in verdicts:
        name = item.get("name") or item.get("role") or "Juez"
        lines.append(
            f"{sub}# {name} — {_score_label(item.get('nota'), scale)}"
        )
        lines.append("")
        lines.append(
            f"- Exposición: {_score_label(item.get('exposicion'), scale)}"
        )
        lines.append(
            f"- Respuestas: {_score_label(item.get('respuestas'), scale)}"
        )
        lines.append("")
        lines.append("**Fortalezas**")
        strengths = item.get("fortalezas") or []
        if strengths:
            lines.extend(f"- {entry}" for entry in strengths)
        else:
            lines.append("- Sin fortalezas registradas.")
        lines.append("")
        lines.append("**Debilidades**")
        weaknesses = item.get("debilidades") or []
        if weaknesses:
            lines.extend(f"- {entry}" for entry in weaknesses)
        else:
            lines.append("- Sin debilidades registradas.")
        lines.append("")
        comment = (item.get("comentario") or "").strip()
        lines.append(f"**Comentario:** {comment or 'Sin comentario.'}")
        if item.get("parse_error"):
            lines.extend(
                [
                    "",
                    "> No se pudo interpretar la evaluación de este juez; "
                    "la nota no cuenta para el promedio.",
                ]
            )
        lines.append("")

    lines.extend([f"{sub} Preguntas y respuestas", ""])
    questions = final.get("questions") or []
    if not questions:
        lines.append("- Sin preguntas registradas.")
    for item in questions:
        kind = _TURN_KIND_MARKDOWN.get(
            str(item.get("kind") or ""), "Pregunta"
        )
        lines.append(
            f"- **{item.get('judge') or item.get('role')}** ({kind}): "
            f"{item.get('question')}"
        )
        lines.append(f"  - Respuesta: {item.get('answer')}")
    return "\n".join(lines).rstrip()
