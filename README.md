# Simulador de jurado — defensa TEG CUAM

> 🇪🇸 **¿Eres estudiante?** Lee la guía en español: [README.es.md](README.es.md).

Local voice app to rehearse the CUAM thesis defense: present slide by slide in
practice mode or in one continuous "defensa real" recording, answer three AI
judges shown as generated avatars seated at a jury table, get per-slide and
global feedback, and ask a tutor any doubt. Everything runs on `127.0.0.1`; no
build step and no CDN, just FastAPI and a plain HTML/CSS/JS page.

## Use it for your own thesis

1. Put your documents in a folder (by default `~/tesis/`) and point
   `config.json` → `sources` at them: `thesis` (your `.docx`), `norms` (your
   institution's evaluation norms as PDF), `deck` (your defense slides as PDF)
   and `study_guide` (any HTML or notes file you study from).
2. Set `student.name` to your name; the judges greet you with it.
3. Adjust the judges in `config.json` → `roles` (names, personas, voices) and
   the shared rules in `rules` if your institution or level differs from the
   CUAM TSU defense this was built for.
4. Install the requirements below and follow the quick path.

Your thesis, recordings and saved practices stay on your machine: `data/` and
`runtime/` are git-ignored.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quick path

```bash
cd jurado-defensa

# 1. Build the context (thesis, CUAM norms, deck slides). Re-run after editing the thesis or deck.
.venv/bin/python scripts/build_context.py
# 1b. Neural voices (Piper, offline): pip install piper-tts in .venv, then
scripts/download_voices.sh

# 2. Run the app: starts an isolated `opencode serve`, whisper STT and `say` TTS
.venv/bin/python -m app.main          # JURADO_PORT=8001 to change the port

# 3. Open the UI (Chrome or Safari; allow the microphone)
open http://127.0.0.1:8000
```

Requirements: OpenCode logged in to the `opencode-go` provider (all roles use
`opencode-go/deepseek-v4.1-flash`), `ffmpeg`, `pdftotext`/`pdftoppm`, and the
a whisper.cpp model, by default `~/Library/Application Support/Jarvis/models/ggml-large-v3-turbo-q5_0.bin`
(download `ggml-large-v3-turbo-q5_0.bin` from the whisper.cpp models on Hugging
Face and set `JURADO_WHISPER_MODEL=/path/to/model.bin` to use any location).

Judges, tutor and evaluator personas, voices and models live in `config.json`
(single source: names, tone, questioning style, favourite topics, `voice`).
`student.name` feeds the jury's opening line (`jury.opening_line`, with a
`{name}` placeholder) and `jury.target_minutes_min` / `jury.target_minutes_max`
(default 15 and 18) drive the real-defense clock warning and the global report.
`jury.interjection_probability` (default `0.35`) is the chance that another
judge interjects after an answer. Judges set `"variant": "low"` (OpenCode
`model.variant` → `reasoningEffort: low`) to cut thinking time; the model is
unchanged for every role. Generated audio and saved practices go to
`data/audio/` and `data/practicas/`. The context build also writes
`data/context/capitulos.json`: the thesis chapters (`CAPÍTULO I`–`V`, plus
`Introducción`) with their line ranges and section titles, used by the jury's
chapter selector.

**Assumption about the grade:** `jury.grading` sets `scale_max: 20`, `pass_mark:
10` and the exposition/answers weights (`0.5`/`0.5`). The Venezuelan 1-20 scale
and a pass mark of 10 are an assumption, not a CUAM rule; change `scale_max`
and `pass_mark` in `config.json` to match your own criteria. The model only
returns the two component scores (0-20); the weighted nota, the mean, the
"Aprobado"/"Reprobado" result and the mention are computed in Python.
`jury.direct_opening_line` is the greeting spoken by the president in the
direct-jury mode (default "Buenas tardes, {name}. Vamos directo a las
preguntas.").

The jury only starts when the exposition has real content: at least
`jury.min_exposition_words` words (default `80`) across all slide transcripts.
Spoken fillers such as `ok`, `eh`, `este`, and `mm` are not counted. The
**Jurado directo** entry bypasses this guard: it marks the session with
`jury_mode: "directo"` and starts the questions without an exposition.

### Practising one chapter at a time

The jury panel has a **Capítulos para las preguntas** selector (loaded from
`GET /api/chapters`): *Todos los capítulos* by default, or any combination of
the thesis chapters (`Introducción`, Capítulo I–V). Both `POST /api/jury/start`
and `POST /api/jury/direct` accept `chapters: ["cap1", ...]`; an empty or
missing list means the whole thesis. Every judge question prompt (first, next
main, follow-up, interjection and direct mode) then names the chosen chapters,
their line ranges and their section titles, and instructs the judge to ask
**only** about that content with its own specialty (for Capítulo I the
technician asks how the problem justifies a software solution, not about the
database design); if the answer drifts to another chapter, the judge steers it
back kindly. The thesis text is not resent: it is already in the shared cached
context prefix. The chosen scope is shown on the jury screen
(*Preguntas sobre: Capítulo I · El problema*), is passed to the final
deliberation (*evaluaste solo …*) and appears in the verdict screen and the
Markdown. Prefetched questions are keyed by the scope, so changing it never
serves a question generated for another chapter.

## Modes

| Mode | What it does |
| --- | --- |
| **1 · Exposición** | Two modes in one screen. **Modo práctica:** shows the current slide image and number, a per-slide timer, and records the microphone with `MediaRecorder` per slide; **Comenzar**, **Regrabar lámina**, **Siguiente lámina**, **Terminar**. **Defensa real:** the president judge speaks the opening line from `config.json` ("Buenas tardes. Adelante, Tu Nombre, ya puedes empezar."), then one continuous recording covers the whole defense with a visible total clock and the slide counter; slide changes only add timestamps. Each slide's audio is uploaded (practice) or the full recording plus its timeline is uploaded once (real) and transcribed in the background; the list below the slide shows `Sin grabar`, `Transcribiendo…`, `Lista`, `No se entendió — repite`, or `Error al transcribir`. |
| **2 · Jurado** | After **Terminar**, the jury table shows the three judges seated behind a simple desk with name plates (metodóloga left, presidenta center and slightly larger, técnico right). The avatars are generated offline (DiceBear Avataaars) and each one has closed, half-open and open mouth frames plus a blink frame. The app decides the turn order: the next main asker is random among judges with budget left and never repeats the previous one while another judge still has questions. `questions_per_judge` is the main-question budget per judge (1 → 3 questions, 2 → 6). After an answer the original asker may ask its single follow-up, and with `jury.interjection_probability` (default 0.35) another judge makes a short interjection about that same question and answer before the next main question; interjections do not consume budget. Each judge receives the full transcript once. Answers are push-to-talk (hold the button or the space bar) or text. When the budgets are used, the jury enters **deliberación**: the three characters show **pensando…** under a "El jurado está deliberando…" banner while every judge grades the whole defense in parallel (full exposition per slide with duration, every question by any judge with your answer, and total time vs. the 15-18 min target). The final grade is computed in Python (per-judge nota = 0.5·exposición + 0.5·respuestas, mean of the three, 0-20 scale from `config.json`), the president reads a short closing aloud with her voice, and the **Veredicto del jurado** screen shows the big final grade, Aprobado/Reprobado + mention, a card per judge (nota, exposición, respuestas, fortalezas, debilidades, comentario), every question with its answer, and a Markdown download. The verdict is also part of the feedback Markdown and of the saved practice JSON. While a question's TTS plays, that character's mouth follows the real audio level (Web Audio `AnalyserNode`) with a subtle glow; idle characters blink and breathe; the judge waiting for your answer shows **Escuchando…**, holding push-to-talk shows **Te escucho…**, and the thinking judge shows a **pensando…** bubble. While the brain thinks after you answer, the asking judge plays one of the four pre-synthesized reaction clips from `GET /api/jury/fillers`. The current question appears in a speech bubble over the asking judge. `prefers-reduced-motion` disables the bounce/blink animations and leaves the mouth static. The jury refuses to start while the exposition is below `jury.min_exposition_words` real words and shows how many words were heard. |
| **Jurado directo** | A shortcut for when you do not want to record an exposition: the **Jurado directo** button on the exposition screen, in the nav, or in the jury panel starts the jury with `jury_mode: "directo"`. There is no minimum-word guard and no transcript; the president greets you with `jury.direct_opening_line` ("Buenas tardes, Tu Nombre. Vamos directo a las preguntas.") and the first judge asks a natural question about the defense deck and the thesis. Warm-up and question prefetch work as in the normal mode. In the deliberation `puntaje_exposicion` is `null`, each judge's nota is the answer score alone, and the verdict screen and Markdown say **Sin exposición (modo directo)**. |
| **3 · Retroalimentación** | Calls the `evaluador` agent once per slide with the slide text plus that slide's transcript. Shows a `dijo` summary, strengths, corrections, omissions, statements that contradict the thesis, a 0-10 score, and the Python-computed muletillas and words per minute. Muletillas are named one by one ("o sea ×3", "digamos", "como que"...). A retake of a slide keeps the previous take in `attempts`, and the evaluator says what improved (`mejoras_vs_anterior`). A **Tu progreso** card shows the jury grade trend, the per-slide average, the slides that keep scoring under 7 (links to practice them) and the filler habits across all saved practices. "Generar retroalimentación" evaluates two slides at a time. Slides with no or filler-only speech are skipped and reported as `Sin exposición suficiente en esta lámina`. A global report on top adds total time vs. the 15-18 minute target, words per minute, total muletillas, the five CUAM Unidad III rubric scores, the 3 top priorities and the 3 weakest slides. The whole report is downloadable as Markdown. |
| **4 · Tutor** | Free chat with the tutor by text or push-to-talk. Replies can optionally be spoken; the tutor avatar animates while its reply is playing. |

### Keyboard shortcuts (Exposición)

| Key | Modo práctica | Defensa real |
| --- | --- | --- |
| `→` / `PageDown` | Same as **Siguiente lámina**: uploads the current slide and records the next one. Before **Comenzar** or after **Terminar** it only browses. | Next slide; the single recording keeps running |
| `←` / `PageUp` | Saves the current take, pauses the recording and shows the previous slide, leaving its earlier transcript untouched (press **Reanudar** to record again). Before **Comenzar** or after **Terminar** it only browses. | Previous slide; going back appends to that slide again |

The on-screen hint **← → para cambiar de lámina** is visible above the slide.
Arrows are ignored while you type in an input or hold `Space` for push-to-talk.
`Space` is reserved for push-to-talk in the jury and tutor tabs and does nothing
in the exposition, so it never stops the real-defense recording.


## How it fits together

| File | Responsibility |
| --- | --- |
| `app/main.py` | FastAPI app factory, JSON API, background transcription, jury turn-taking, evaluator parsing, tutor chat, static UI. |
| `app/session.py` | In-memory practice-session state, judge metadata, slide timeline, JSON persistence under `data/practicas/`, feedback report and Markdown rendering. |
| `app/stt.py` | ffmpeg normalization plus whisper transcription, text only (`transcribe_audio`) or with timestamps (`transcribe_segments`). |
| `web/` | Single-page UI (`index.html`, `styles.css`, `app.js`). No dependencies. `web/characters.js` layers the four generated SVG frames per character and exposes the mouth/activity hooks used by `app.js`. |

`create_app` is dependency-injected so the app can be tested with fakes:

```python
create_app(
    brain,               # create_session(role) -> str; ask(session_id, text) -> str
    stt,                 # transcribe_audio(path) -> str; transcribe_segments(path) -> [(start_s, end_s, text)]
    tts,                 # synthesize(text, role, out_dir) -> Path (wav)
    slides_dir,          # directory with slide-NN.png / slide-NN.txt
    audio_dir,           # generated wavs and uploaded recordings
    session_dir=None,    # defaults to <audio_dir>/../practicas
    min_exposition_words=80,  # main() reads jury.min_exposition_words from config.json
    student_name="Tu Nombre",  # config.json student.name
    opening_line="Buenas tardes. Adelante, {name}, ya puedes empezar.",  # jury.opening_line
    target_minutes_min=15,  # jury.target_minutes_min
    target_minutes_max=18,  # jury.target_minutes_max
    interjection_probability=0.35,  # jury.interjection_probability
    rng=None,            # injectable random.Random for deterministic turn order in tests
    tutor_name="Prof. Andrés Salas",  # config roles.tutor.name
    executor=None,       # concurrent.futures.Executor for warm-up/prefetch (tests inject one)
    warmup_enabled=True,  # tiny ask per judge when the exposition starts
    prefetch_enabled=True,  # pre-generate the next main questions
    filler_phrases=(...),  # neutral reaction clips; () disables them
    prepare_fillers_on_startup=False,  # main() sets True for the real server
    grading=None,        # jury.grading from config.json (scale, pass mark, weights)
    direct_opening_line="Buenas tardes, {name}. Vamos directo a las preguntas.",  # jury.direct_opening_line
    deliberation_executor=None,  # concurrent.futures.Executor for the S15 deliberation (tests inject one)
    chapters=None,       # data/context/capitulos.json (chapter selector); [] disables it
)
```

Roles passed to `brain.create_session`: `juez_metodologo`, `juez_tecnico`,
`juez_presidente`, `evaluador`, `tutor`. The evaluator is expected to answer
per slide with JSON like:

```json
{"slide": 1, "dijo": "...", "fortalezas": ["..."], "correcciones": ["..."],
 "omisiones": ["..."], "errores_vs_tesis": ["..."], "frase_sugerida": "...",
 "puntaje": 8}
```

and, for `POST /api/feedback/global`, with:

```json
{"rubrica": {"dominio_del_tema": {"puntaje": 8, "comentario": "..."}},
 "prioridades": ["...", "...", "..."], "laminas_debiles": [4, 12, 20]}
```

The parser accepts plain JSON, JSON inside a Markdown fence, or JSON embedded in
prose, and falls back to `parse_error: true` plus the raw reply when nothing can
be decoded. `muletillas` and `palabras_por_minuto` are always computed in Python
from the transcript and duration, never taken from the model.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Single-page UI |
| `GET` | `/api/health` | Health check and slide count |
| `GET` | `/api/slides` | Slide list (number, text, image URL) |
| `GET` | `/api/slides/{n}/image` | Slide image (`?v=<mtime>` of the current file is cached as immutable) |
| `GET` | `/api/progress` | Summary of saved practices: grades, weak slides, filler habits |
| `GET` | `/api/exposition/{n}/status` | Light transcription status of one slide (practice polling) |
| `POST` | `/api/session` | Start a new practice (optional `questions_per_judge`) |
| `GET` | `/api/session` | Current session state |
| `POST` | `/api/exposition/start` | Mark the exposition as started (optional `{"mode": "practice"\|"real"}`) |
| `POST` | `/api/exposition/opening` | Speak the jury's opening line with the president judge's voice |
| `POST` | `/api/exposition/{n}/audio` | Upload one slide's recording (`file`, optional `duration`); transcription runs in the background |
| `POST` | `/api/exposition/full/audio` | Upload one continuous defense recording (`file`, `timeline` JSON list of `{slide, start_ms}`, optional `duration`); transcription runs once and is split by segment midpoint |
| `POST` | `/api/session/finish` | Finish the exposition and save the session JSON |
| `POST` | `/api/jury/start` | Reuse/create the judge brain sessions and ask the first question (prefetched when ready) |
| `POST` | `/api/jury/direct` | Start the jury without an exposition (Jurado directo) |
| `GET` | `/api/jury/fillers` | Four pre-synthesized reaction clips per judge (`{role: [audio_url, ...]}`) |
| `POST` | `/api/jury/answer` | Answer with text |
| `POST` | `/api/jury/answer/audio` | Answer with audio (transcribed server-side) |
| `GET` | `/api/jury/verdict/markdown` | Download the final jury verdict as Markdown (S15) |
| `POST` | `/api/feedback/{n}` | Evaluate one slide |
| `POST` | `/api/feedback` | Evaluate every pending slide |
| `POST` | `/api/feedback/global` | Generate the global report (rubric, priorities, weakest slides) |
| `GET` | `/api/feedback` | Feedback report |
| `GET` | `/api/feedback/markdown` | Download the report as Markdown |
| `GET` | `/api/tutor` | Tutor history |
| `POST` | `/api/tutor/message` | Tutor question by text (optional `speak`) |
| `POST` | `/api/tutor/audio` | Tutor question by audio |
| `GET` | `/api/audio/{name}` | Generated wav audio |
| `GET` | `/api/audio/fillers/{name}` | Pre-synthesized filler clip |

The jury JSON keeps its existing fields and adds `current_role`,
`turn_kind` (`pregunta`, `repregunta`, `intervencion`) and a `remaining`
main-question budget per judge. Each turn also carries its `kind`. The next
main asker is chosen with `random.Random.choice` from the judges with budget,
excluding the previous asker when possible; `create_app` accepts an injectable
`rng` so tests can pin the order.

Brain calls run in a worker thread (`run_in_threadpool`), so the event loop
keeps serving the UI while a judge or the evaluator is thinking. The UI shows a
`pensando…` pill during those waits.

## Latency

The jury is meant to feel conversational, so the slow parts run while the
student is not waiting:

- **Shared cache prefix.** Every agent prompt starts with the same context block
  (`tesis`, `normas`, `presentacion`, `guia_estudio`, same order and bytes); the
  role persona and rules come after it. All judge, evaluator and tutor sessions
  then share one provider cache prefix.
- **Warm-up.** When the exposition starts (practice or real), the app creates
  the three judge sessions and sends one tiny background ask per judge
  (`"Responde solo: listo"`). The request that starts the exposition never waits
  for it; failures are logged and ignored.
- **Prefetch.** When the exposition finishes, the app pre-generates the first
  main question of every judge in parallel, TTS included. After each answer it
  keeps the answered judge's next main question ready while the student talks to
  the next judge. A prefetched question is consumed once; if it is not ready
  yet, the request waits for it instead of asking again. Answer-dependent turns
  (follow-up, interjection, verdict) are always generated live.
- **Reasoning effort.** The judges use `"variant": "low"` in `config.json`
  (OpenCode `model.variant`), which maps to `reasoningEffort: low`; the model
  itself is unchanged.
- **Fillers.** At startup the app synthesizes four neutral reaction clips per
  judge with its own voice into `data/audio/fillers/`; existing clips are reused
  and a failing voice is skipped.
- **Sentence streaming.** A judge reply is split into sentence chunks (the
  opening reaction stands alone; a long opening sentence is cut at a clause or
  space so the first chunk stays short). The API returns the question text and
  `jury.audio_chunks` as soon as the brain answers; chunk 1 is synthesized first
  and the rest in the background. `GET /api/audio/<name>` waits (bounded, 20 s)
  for a pending chunk, and the UI plays the chunks back to back with the mouth
  animation, switching the character from "pensando…" to "Hablando…".
- **Speech priority.** All synthesis runs through one serial priority queue
  (`PRIORITY_HIGH` live turn, `PRIORITY_NORMAL` one-offs, `PRIORITY_LOW`
  prefetch/fillers) in front of the single XTTS worker. Consuming a prefetched
  question promotes its pending chunks instead of synthesizing them twice.
- **Deadlines.** Each turn has a `config.json jury.deadlines` budget
  (`pregunta`/`repregunta` 25 s, `intervencion` 12 s, `veredicto` 60 s). On
  expiry the app calls `POST /api/session/<id>/interrupt`, logs
  `brain deadline role=… kind=… after=…s action=…` and retries once in a fresh
  session; a follow-up is treated as answered, an interjection is skipped, a
  main question moves to the next judge, and the deliberation grades with the
  judges that answered. The student sees "El juez no respondió a tiempo,
  seguimos." instead of a stuck "pensando…".
- **Timings.** Every turn records `stt_ms`, `brain_ms`, `tts_ms`,
  `first_audio_ms`, `wait_ms` and `prefetched` in `jury.timings` and logs
  `jury turn role=… kind=… … wait_ms=…` plus a `jury turn audio …` line when
  chunk 1 is on disk. `wait_ms` is answer request → response; `first_audio_ms`
  is answer request → chunk 1 ready (0 when a prefetched question was already
  synthesized).

## Voice comparison (TTS listening test)

The judges' voices are configurable through `config.json` (`voice`, either a
`piper:<model>#<speaker>` id or a macOS `say` name). Before switching engines,
`scripts/voice_samples.py` builds a listening comparison with the same two lines
per jury role: Piper (the current baseline), Kokoro-82M and XTTS-v2. It writes
`data/voice-samples/<engine>/<role>-<n>.wav`, a `timings.json`, and
`data/voice-samples/index.html`, a static page (relative paths, opens directly
in Chrome) with one `<audio>` player per engine plus the synthesis seconds and
the real-time factor (RTF: compute seconds per audio second) under each.

The engines live outside the app `.venv`, one venv per engine under
`~/Library/Application Support/JuradoDefensa/engines/<engine>/`, because their
dependencies conflict (torch for XTTS, onnxruntime for Kokoro):

```bash
# one-time install (Python 3.12)
engines="$HOME/Library/Application Support/JuradoDefensa/engines"
python3.12 -m venv "$engines/kokoro/venv"
"$engines/kokoro/venv/bin/pip" install kokoro-onnx   # model+voices download on first run
python3.12 -m venv "$engines/xtts/venv"
"$engines/xtts/venv/bin/pip" install "coqui-tts[codec]" torch torchaudio six

# regenerate every sample and the page (data/ is gitignored)
.venv/bin/python scripts/voice_samples.py            # --engine piper|kokoro|xtts for one
open data/voice-samples/index.html
```

Kokoro v1.0 ships only one Spanish female voice (`ef_dora`) and two male ones
(`em_alex`, `em_santa`); the second female sample for the presidenta is an
`ef_dora + af_heart` blend (80/20 style vectors) that must be auditioned by ear.
`scripts/voice_samples.py --list-voices` prints the 58 XTTS-v2 studio speakers;
the four used here were chosen by median F0 and speaking rate over the Spanish
opening line. The XTTS-v2 weights are licensed CPML (non-commercial), which is
fine for this personal rehearsal; the app never downloads or ships them.

## Characters

The jury and tutor avatars are **static SVG files** under `web/characters/`
(`<role>/<emotion>/closed.svg`, `half.svg`, `open.svg`, `blink.svg`), generated once with
[DiceBear](https://www.dicebear.com) `@dicebear/core` + `@dicebear/collection`
and the **Avataaars** style by Pablo Stanley (free for personal and commercial
use; see `web/characters/LICENSE.md`). No CDN or build step runs at runtime;
FastAPI serves them from `/static/characters/` like the rest of `web/`.

Every character has seven emotions: `neutral`, `feliz`, `enojado`,
`esceptico`, `sorprendido`, `pensativo` and `preocupado`. Each one changes the
eyebrows, eyes and resting mouth, tints the halo behind the head, sets a pose
(lean, tilt, droop), shows a floating symbol (sparkle, anger mark, `?`, `!`,
`…`, sweat drop) and plays a one-shot reaction when it starts (shake, hop,
pop, droop). The mood is chosen in the browser: a judge's question is read by
`emotionFromText` (cues such as "no respondiste" or "contradice" → enojado,
"me preocupa" → preocupado, "¿está seguro…?" → escéptico; a follow-up defaults
to escéptico), deliberation is `pensativo`, and a verdict or a practice score
maps to an emotion through `emotionFromScore`. The jury stage is built once and
updated in place, so polling never restarts the animations.

While the student answers, the judges react live (`listeningEmotion`), each
with its own temperament: the methodologist turns skeptical on a pause and
upset on a long silence (6 s) or an answer over 90 s; the technical judge goes
pensive and curious; the president smiles and gets worried. Signals are
computed in the browser only: the voice level of the push-to-talk microphone
(Web Audio analyser, nothing is sent anywhere) and, for typed answers, the
words so far (`answerMood`: "no sé", "creo que" → weak; "según la tesis", data
→ strong). Faces change at most every 1.8 s.

`web/characters.js` stacks the four frames of the current emotion in one
portrait and shows only the active mouth state. `app.js` keeps calling `setMouth(root, level)` with the RMS
level of the playing `<audio>`; the closed/half/open frames switch when the
level crosses a threshold. Blinking uses the `blink` frame through a CSS
animation, breathing/bounce/glow come from CSS, and `prefers-reduced-motion`
disables the animations.

To regenerate the assets, install the dependencies in a throwaway directory
(outside this repository) and run the committed script there:

```bash
repo="$PWD"
tmp="$(mktemp -d)"
cp scripts/gen_characters.mjs "$tmp/"
(cd "$tmp" && pnpm add @dicebear/core@9 @dicebear/collection@9)
(cd "$tmp" && node gen_characters.mjs "$repo/web/characters")
rm -rf "$tmp"
```

Each role's fixed look (hair, skin tone, glasses, beard, clothing) lives in the
`ROLES` table inside the script, so regenerating is deterministic.

## Requirements

| Piece | Expectation |
| --- | --- |
| Python | 3.12 virtualenv at `.venv` (fastapi, uvicorn, python-multipart, httpx, pytest). |
| `ffmpeg` | Converts browser recordings (`webm`/`mp4`) to 16 kHz mono wav before STT. Without it the original file is passed through. |
| Microphone | Browser permission for `localhost`; Chrome, Edge, or Safari. |
| Brain / STT / TTS | Wired in `main()` from `app/brain.py`, `app/stt.py`, `app/tts.py` (T1 and T2). |

## Tests

```bash
.venv/bin/pytest -q
```

The tests use `FastAPI.TestClient` and deterministic fakes for brain and TTS,
plus real `say`-generated clips for whisper STT. They cover slide listing, audio
upload and stored transcripts, the real-defense timeline (midpoint assignment,
silent slides, returning to a previous slide), the opening line, the dynamic
jury turn order (seeded random asker without repeats, exact main budget per
judge, follow-ups, interjections at probability 1.0/0.0 that do not consume
budget, verdicts after the budgets), per-slide and global evaluator parsing
including malformed JSON, Python-computed muletillas and words per minute,
tutor chat, session persistence, and the static UI (including the character
assets and reduced-motion CSS). S13 adds the byte-identical shared prompt prefix
across roles, the model variant on judge sessions, one non-blocking warm-up ask
per judge, a prefetched first question that is consumed exactly once, the filler
clips endpoint, and the per-turn timings in the jury JSON and the log. S15 adds
the deliberation prompt (whole exposition + every question/answer + total
time), the Python grade math and mention thresholds, the configurable pass mark,
the defensive parsing of a malformed judge reply, the president's closing TTS,
the verdict UI and Markdown export, and the arrow-key slide navigation in both
exposition modes. The direct-jury tests cover the skipped exposition guard, the
prompt without a transcript, the warm-up without an exposition, and the grade
that uses the answers only.

## Notes

- One practice session is active at a time; **Nueva práctica** replaces it.
- Session JSON files are written when the exposition finishes, after the jury
  completes, and after feedback generation. They contain the transcripts, so
  `data/` is gitignored.
- Uploaded audio is stored under `audio_dir`; generated judge/tutor wavs live
  there too and are served by `/api/audio/{name}`.
- The judge names and personas displayed in the UI come from `config.json`
  (`roles`) through `configure_judges`; the `JUDGES` defaults in
  `app/session.py` are only fallbacks for tests. The generated avatar for each
  role is keyed by the same role name and matches those configured people
  (Yajaira Contreras: dark hair up, glasses, formal blazer; Jesús Rondón:
  younger man with a short beard in a casual shirt; Carmen Rojas: silver hair,
  elegant blazer, center seat; Andrés Salas: glasses and grey cardigan).
