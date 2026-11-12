"""Character artwork: every role has every emotion, and moods are read sensibly."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
ROLES = ["juez_metodologo", "juez_tecnico", "juez_presidente", "tutor"]
EMOTIONS = ["neutral", "feliz", "enojado", "esceptico", "sorprendido", "pensativo", "preocupado"]
FRAMES = ["closed", "half", "open", "blink"]


def run_characters(expression: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = (
        f"const c = require({json.dumps(str(WEB / 'characters.js'))});"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("role", ROLES)
def test_every_role_has_all_emotion_frames(role):
    for emotion in EMOTIONS:
        for frame in FRAMES:
            path = WEB / "characters" / role / emotion / f"{frame}.svg"
            assert path.is_file(), path
            assert path.read_text(encoding="utf-8").lstrip().startswith("<svg"), path


def test_frames_of_one_emotion_differ_only_where_they_should():
    folder = WEB / "characters" / "tutor"
    assert (folder / "enojado" / "closed.svg").read_text() != (
        folder / "feliz" / "closed.svg"
    ).read_text()
    assert (folder / "neutral" / "closed.svg").read_text() != (
        folder / "neutral" / "open.svg"
    ).read_text()


def test_script_lists_the_same_emotions_as_the_assets():
    assert run_characters("c.emotions") == EMOTIONS


def test_figure_uses_the_emotion_folder_and_falls_back_to_neutral():
    html = run_characters('c.figure("juez_tecnico", "enojado")')
    assert "/static/characters/juez_tecnico/enojado/open.svg" in html
    assert 'data-emotion="enojado"' in html
    assert "molesto" in html
    fallback = run_characters('c.figure("juez_tecnico", "furioso")')
    assert "/static/characters/juez_tecnico/neutral/closed.svg" in fallback


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        ("No respondiste lo que te pregunté.", "pregunta", "enojado"),
        ("Eso contradice lo que dice su tesis.", "intervencion", "enojado"),
        ("Me preocupa que la muestra sea tan pequeña.", "pregunta", "preocupado"),
        ("Excelente, muy bien explicado.", "pregunta", "feliz"),
        ("Buenas tardes, Ana. Vamos directo a las preguntas.", None, "feliz"),
        ("¿Está seguro de que esa es la población?", "pregunta", "esceptico"),
        ("Qué interesante ese enfoque.", "pregunta", "sorprendido"),
        ("¿Cuál es el objetivo general?", "pregunta", "neutral"),
        ("¿Cuál es el objetivo general?", "repregunta", "esceptico"),
        ("¿Y el cronograma?", "intervencion", "sorprendido"),
    ],
)
def test_emotion_from_text(text, kind, expected):
    assert run_characters(f"c.emotionFromText({json.dumps(text)}, {json.dumps(kind)})") == expected


@pytest.mark.parametrize(
    ("score", "scale", "expected"),
    [(18, 20, "feliz"), (13, 20, "neutral"), (10.5, 20, "preocupado"), (6, 20, "enojado"),
     (9, 10, "feliz"), (None, 20, "neutral")],
)
def test_emotion_from_score(score, scale, expected):
    assert run_characters(f"c.emotionFromScore({json.dumps(score)}, {scale})") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No sé, creo que eran unos cinco", "weak"),
        ("Según la tesis la muestra fue de 30 personas", "strong"),
        ("El sistema registra los productos", None),
        ("no sé", None),
    ],
)
def test_answer_mood(text, expected):
    assert run_characters(f"c.answerMood({json.dumps(text)})") == expected


def listening(role, **signals):
    return run_characters(f"c.listeningEmotion({json.dumps(role)}, {json.dumps(signals)})")


def test_long_silence_upsets_the_strict_judge_and_worries_the_president():
    assert listening("juez_metodologo", elapsed=20, silence=7) == "enojado"
    assert listening("juez_presidente", elapsed=20, silence=7) == "preocupado"
    assert listening("juez_tecnico", elapsed=20, silence=7) == "esceptico"


def test_short_pause_and_weak_words_change_the_faces():
    assert listening("juez_metodologo", elapsed=10, silence=3) == "esceptico"
    assert listening("juez_tecnico", elapsed=10, silence=3) == "pensativo"
    assert listening("juez_metodologo", elapsed=10, silence=0, mood="weak") == "enojado"
    assert listening("juez_presidente", elapsed=10, silence=0, mood="strong") == "feliz"


def test_long_answers_make_judges_impatient():
    assert listening("juez_metodologo", elapsed=95, silence=0) == "enojado"
    assert listening("juez_presidente", elapsed=95, silence=0) == "preocupado"


def test_attentive_faces_cycle_over_time_and_differ_between_judges():
    faces = [listening("juez_presidente", elapsed=t, silence=0) for t in (0, 5, 10, 15)]
    assert faces == ["neutral", "pensativo", "neutral", "feliz"]
    strict = [listening("juez_metodologo", elapsed=t, silence=0) for t in (0, 5, 10, 15)]
    curious = [listening("juez_tecnico", elapsed=t, silence=0) for t in (0, 5, 10, 15)]
    assert strict != curious != faces
    assert listening("tutor", elapsed=0, silence=10) == "neutral"
