"use strict";

const ROLE_LABELS = {
  juez_metodologo: "Metodóloga",
  juez_tecnico: "Técnico",
  juez_presidente: "Presidenta del jurado",
};

const JURY_LAYOUT = ["juez_metodologo", "juez_presidente", "juez_tecnico"];

const TURN_KIND_LABELS = {
  pregunta: "Pregunta",
  repregunta: "Repregunta",
  intervencion: "Intervención",
};

const STATUS_LABELS = {
  pending: "Sin grabar",
  transcribing: "Transcribiendo…",
  done: "Lista",
  empty: "No se entendió — repite",
  error: "Error al transcribir",
};

const state = {
  session: null,
  slides: [],
  slideIndex: 0,
  tab: "exposition",
  recording: null,
  expoRunning: false,
  expoMode: "practice",
  realStarted: false,
  recordingStartMs: null,
  timeline: [],
  overTimeWarned: false,
  openingUrl: null,
  busy: false,
  juryBusy: false,
  juryClosingPlayed: false,
  awaitingClosing: false,
  pttHeld: false,
  tutorBusy: false,
  report: null,
  global: null,
  // S21: one slide at a time with the tutor's feedback.
  practice: {
    index: 0,
    busy: false,
    startedAt: null,
    timer: null,
    audioUrl: null,
    attempts: {},
  },
  // S22: images waiting to be sent to the tutor.
  tutorImages: [],
};

let slideTimer = null;
let slideStartedAt = null;
let pollTimer = null;
let thinkingCount = 0;
let errorTimer = null;

// ---------------------------------------------------------------------------
// DOM and formatting helpers
// ---------------------------------------------------------------------------

function $(id) {
  return document.getElementById(id);
}

// Named filler habits ("o sea ×3"), most frequent first.
function renderMuletillas(breakdown) {
  const entries = Object.entries(breakdown || {});
  if (!entries.length) return "";
  const chips = entries
    .map(
      ([word, count]) =>
        `<span class="chip">${escapeHtml(word)} <strong>×${escapeHtml(count)}</strong></span>`
    )
    .join("");
  return `<p class="muletillas">Tus muletillas: ${chips}</p>`;
}

function escapeHtml(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatSeconds(value) {
  const total = Math.max(0, Math.floor(Number(value) || 0));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

const FILLER_WORDS = new Set([
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
]);

function expositionWordCount(value) {
  const words = String(value || "").toLowerCase().match(/[\p{L}]+/gu) || [];
  return words.filter((word) => !FILLER_WORDS.has(word)).length;
}

// ---------------------------------------------------------------------------
// API and user feedback
// ---------------------------------------------------------------------------

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options || {});
  } catch (_) {
    throw new Error("No se pudo conectar con el servidor local.");
  }
  if (!response.ok) {
    let detail = `Error ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.detail) {
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch (_) {
      /* non-JSON error body */
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

function showError(message) {
  const banner = $("error-banner");
  banner.textContent = message;
  banner.classList.remove("notice");
  banner.hidden = false;
  clearTimeout(errorTimer);
  errorTimer = setTimeout(() => {
    banner.hidden = true;
  }, 9000);
}

function showNotice(message) {
  const banner = $("error-banner");
  banner.textContent = message;
  banner.classList.add("notice");
  banner.hidden = false;
  clearTimeout(errorTimer);
  errorTimer = setTimeout(() => {
    banner.hidden = true;
    banner.classList.remove("notice");
  }, 7000);
}

function setThinking(active, message) {
  thinkingCount = Math.max(0, thinkingCount + (active ? 1 : -1));
  const pill = $("thinking");
  if (message) {
    pill.textContent = message;
  }
  pill.hidden = thinkingCount === 0;
}

async function withThinking(message, task) {
  setThinking(true, message);
  try {
    return await task();
  } finally {
    setThinking(false);
  }
}

function audioFilename(prefix, blob) {
  const type = (blob && blob.type) || "";
  const extension = type.includes("mp4") ? "m4a" : "webm";
  return `${prefix}.${extension}`;
}

// ---------------------------------------------------------------------------
// Character animation (mouth from the real audio level)
// ---------------------------------------------------------------------------

const characters = window.JuryCharacters || {
  figure: () => "",
  setMouth: () => {},
  setActivity: () => {},
  defaultName: (role) => ROLE_LABELS[role] || role,
};

let audioEngine = null;

function reducedMotion() {
  return Boolean(
    window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function audioContext() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) {
    return null;
  }
  if (!audioEngine) {
    audioEngine = { context: new Ctx() };
  }
  if (audioEngine.context.state === "suspended") {
    audioEngine.context.resume().catch(() => {});
  }
  return audioEngine.context;
}

function unlockAudio() {
  if (!reducedMotion()) {
    audioContext();
  }
}

function characterRoot(role) {
  if (!role) {
    return null;
  }
  return document.querySelector(`.seat[data-role="${role}"]`);
}

/*
 * Animate one character while its TTS audio plays. The mouth follows the
 * AnalyserNode level; when Web Audio cannot route the element (blocked
 * context, unsupported browser) a synthetic pulse keeps the character alive.
 */
function speakCharacter(audio, role) {
  const root = characterRoot(role);
  let frame = null;
  if (!root) {
    return () => {};
  }
  characters.setActivity(root, "speaking");
  const reduce = reducedMotion();
  let analyser = null;
  let live = false;
  const context = reduce ? null : audioContext();
  if (context && context.state === "running") {
    try {
      const source = context.createMediaElementSource(audio);
      analyser = context.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.55;
      source.connect(analyser);
      analyser.connect(context.destination);
      live = true;
    } catch (_) {
      analyser = null;
      live = false;
    }
  }
  if (reduce) {
    characters.setMouth(root, 0.35);
  } else if (live && analyser) {
    const data = new Uint8Array(analyser.fftSize);
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let index = 0; index < data.length; index += 1) {
        const value = (data[index] - 128) / 128;
        sum += value * value;
      }
      const rms = Math.sqrt(sum / data.length);
      characters.setMouth(root, Math.min(1, rms * 4));
      frame = requestAnimationFrame(tick);
    };
    tick();
  } else {
    const started = performance.now();
    const tick = () => {
      const elapsed = (performance.now() - started) / 1000;
      const level = 0.3 + 0.34 * Math.abs(Math.sin(elapsed * 9));
      characters.setMouth(root, level);
      frame = requestAnimationFrame(tick);
    };
    tick();
  }
  return () => {
    if (frame) {
      cancelAnimationFrame(frame);
    }
    characters.setMouth(root, 0);
    if (root.dataset.activity === "speaking") {
      characters.setActivity(root, "idle");
    }
  };
}

function playCharacterAudio(url, role) {
  if (!url) {
    return null;
  }
  const audio = new Audio(url);
  const stopSpeaking = speakCharacter(audio, role);
  audio.addEventListener("ended", stopSpeaking, { once: true });
  audio.addEventListener("error", stopSpeaking, { once: true });
  const attempt = audio.play();
  if (attempt && attempt.catch) {
    attempt.catch(stopSpeaking);
  }
  return audio;
}

/*
 * S19: a judge question streams as sentence chunks. Each URL may still be
 * synthesizing on the server; its GET waits (bounded) for the file, so the
 * chunks are simply played back to back with one continuous mouth animation.
 */
let judgeSpeech = null;

function stopJudgeSpeech() {
  const current = judgeSpeech;
  judgeSpeech = null;
  if (current) {
    current.stop();
  }
}

function juryAudioUrls(jury) {
  if (!jury) {
    return [];
  }
  const chunks = Array.isArray(jury.audio_chunks)
    ? jury.audio_chunks.filter((url) => typeof url === "string" && url.length)
    : [];
  if (chunks.length) {
    return chunks;
  }
  return jury.current_audio_url ? [jury.current_audio_url] : [];
}

function setSeatSpeaking(role, active) {
  const seat = characterRoot(role);
  if (!seat) {
    return;
  }
  const label = seat.querySelector(".seat-state");
  if (label) {
    label.textContent = active ? "Hablando…" : "Escuchando…";
  }
}

function playCharacterAudioList(urls, role) {
  const list = (Array.isArray(urls) ? urls : []).filter(
    (url) => typeof url === "string" && url.length > 0
  );
  if (!list.length) {
    return null;
  }
  stopJudgeSpeech();
  let stopped = false;
  let audio = null;
  let stopMouth = null;
  const finish = () => {
    if (stopped) {
      return;
    }
    stopped = true;
    if (stopMouth) {
      stopMouth();
    }
    if (judgeSpeech && judgeSpeech.audio === audio) {
      judgeSpeech = null;
    }
    setSeatSpeaking(role, false);
  };
  const playAt = (index) => {
    if (stopped) {
      return;
    }
    if (index >= list.length) {
      finish();
      return;
    }
    audio = new Audio(list[index]);
    if (stopMouth) {
      stopMouth();
    }
    stopMouth = speakCharacter(audio, role);
    audio.addEventListener("ended", () => playAt(index + 1), { once: true });
    audio.addEventListener("error", () => playAt(index + 1), { once: true });
    const attempt = audio.play();
    if (attempt && attempt.catch) {
      attempt.catch(() => playAt(index + 1));
    }
  };
  judgeSpeech = {
    audio: null,
    stop() {
      stopped = true;
      if (audio) {
        try {
          audio.pause();
        } catch (_) {
          /* already stopped */
        }
      }
      if (stopMouth) {
        stopMouth();
      }
      setSeatSpeaking(role, false);
    },
  };
  setSeatSpeaking(role, true);
  playAt(0);
  return judgeSpeech;
}

// ---------------------------------------------------------------------------
// Short spoken fillers while a judge thinks
// ---------------------------------------------------------------------------

/*
 * The server exposes GET /api/jury/fillers -> {role: [audio_url, ...]}.
 * The endpoint is optional: a 404 (or any failure) simply disables fillers
 * for this session and never surfaces an error banner. The map is cached
 * after the first successful load.
 */
let fillersCache = null;
let fillersRequest = null;
let judgeFiller = null;
let fillerToken = 0;

function loadJuryFillers() {
  if (fillersCache) {
    return Promise.resolve(fillersCache);
  }
  if (!fillersRequest) {
    fillersRequest = api("/api/jury/fillers")
      .then((body) => {
        const map = {};
        Object.entries(body || {}).forEach(([role, urls]) => {
          if (!Array.isArray(urls)) {
            return;
          }
          const usable = urls.filter(
            (url) => typeof url === "string" && url.length > 0
          );
          if (usable.length) {
            map[role] = usable;
          }
        });
        fillersCache = map;
        return map;
      })
      .catch(() => null)
      .finally(() => {
        fillersRequest = null;
      });
  }
  return fillersRequest;
}

function stopJudgeFiller() {
  const current = judgeFiller;
  judgeFiller = null;
  if (!current) {
    return;
  }
  try {
    current.audio.pause();
  } catch (_) {
    /* already stopped */
  }
  current.stop();
}

async function playJudgeFiller(role, token) {
  if (!role) {
    return;
  }
  const fillers = await loadJuryFillers();
  // The brain may have answered while the filler map was loading, or the
  // turn may have moved on; never start a stale filler over the next audio.
  if (!fillers || !state.juryBusy || token !== fillerToken) {
    return;
  }
  const urls = fillers[role] || [];
  if (!urls.length) {
    return;
  }
  const url = urls[Math.floor(Math.random() * urls.length)];
  stopJudgeFiller();
  const audio = new Audio(url);
  const stopSpeaking = speakCharacter(audio, role);
  const finish = () => {
    if (judgeFiller && judgeFiller.audio === audio) {
      judgeFiller = null;
    }
    stopSpeaking();
  };
  audio.addEventListener("ended", finish, { once: true });
  audio.addEventListener("error", finish, { once: true });
  judgeFiller = { audio, stop: finish };
  const attempt = audio.play();
  if (attempt && attempt.catch) {
    attempt.catch(finish);
  }
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

function currentSlide() {
  return state.slides[state.slideIndex];
}

async function loadInitialSession() {
  let session;
  try {
    session = await api("/api/session");
  } catch (error) {
    if (error.status !== 404) {
      throw error;
    }
    session = await api("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
  }
  applySession(session);
}

function applySession(session) {
  state.session = session;
  state.slides = session.slides || [];
  const index = state.slides.findIndex(
    (slide) => slide.number === session.current_slide
  );
  state.slideIndex = index >= 0 ? index : 0;
  state.expoMode = session.exposition_mode === "real" ? "real" : "practice";
  if ($("expo-mode")) {
    $("expo-mode").value = state.expoMode;
  }
  state.timeline = session.slide_timeline || [];
  state.recordingStartMs = null;
  state.realStarted = false;
  state.overTimeWarned = false;
  state.openingUrl = null;
  if ($("opening-panel")) {
    $("opening-panel").hidden = true;
  }
  state.expoRunning = false;
  state.report = null;
  state.global = session.global || null;
  state.juryClosingPlayed = false;
  state.awaitingClosing = false;
  stopSlideTimer();
  renderAll();
}

async function newSession() {
  if (!window.confirm("¿Empezar una práctica nueva? Se perderá el estado actual.")) {
    return;
  }
  try {
    const session = await withThinking("Creando práctica…", () =>
      api("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      })
    );
    await stopMic();
    applySession(session);
    switchTab("exposition");  } catch (error) {
    showError(error.message);
  }
}

function renderAll() {
  renderSessionBar();
  renderExposition();
  renderSlideStatuses();
  updateExpoButtons();
  renderJury();
  renderPractice();
  renderTutor();
  if (state.report) {
    renderFeedback();
  } else {
    $("feedback-export").hidden = true;
    $("feedback-report").innerHTML = "";
    $("feedback-global").innerHTML = "";
  }
}

function renderSessionBar() {
  const pill = $("session-status");
  if (!state.session) {
    pill.textContent = "Sin práctica";
    return;
  }
  const total = state.slides.length;
  const recorded = state.slides.filter((slide) => slide.has_audio).length;
  const finished = state.session.exposition_status === "finished";
  const recording =
    state.recording && state.recording.owner === "exposition" ? " · grabando" : "";
  pill.textContent = finished
    ? `Práctica terminada · ${recorded}/${total} láminas`
    : `Práctica en curso · ${recorded}/${total} láminas${recording}`;
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function switchTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
    if (button.dataset.tab) {
      button.setAttribute("aria-selected", String(button.dataset.tab === name));
    }
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.hidden = panel.id !== `tab-${name}`;
  });
  if (name === "exposition") {
    renderExposition();
  } else if (name === "jury") {
    renderJury();
  } else if (name === "feedback") {
    loadFeedback();
  } else if (name === "tutor") {
    renderTutor();
  } else if (name === "practice") {
    renderPractice();
  }
}

// ---------------------------------------------------------------------------
// Microphone recording
// ---------------------------------------------------------------------------

async function startMic(owner) {
  const recordingOwner = owner || "exposition";
  if (state.recording) {
    if (state.recording.owner !== recordingOwner) {
      throw new Error(
        "Hay una grabación en curso. Termina la exposición antes de usar el micrófono en otro modo."
      );
    }
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Este navegador no permite grabar audio. Usa Chrome, Edge o Safari actualizado."
    );
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (_) {
    throw new Error(
      "No se pudo acceder al micrófono. Revisa los permisos del navegador y vuelve a intentarlo."
    );
  }
  let options = {};
  if (
    window.MediaRecorder &&
    MediaRecorder.isTypeSupported &&
    MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
  ) {
    options = { mimeType: "audio/webm;codecs=opus" };
  }
  let recorder;
  try {
    recorder = new MediaRecorder(stream, options);
  } catch (_) {
    stream.getTracks().forEach((track) => track.stop());
    throw new Error("No se pudo iniciar la grabación en este navegador.");
  }
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  });
  recorder.start();
  state.recording = { recorder, stream, chunks, owner: recordingOwner };
  renderSessionBar();
  updateExpoButtons();
}

function stopMic() {
  const recording = state.recording;
  if (!recording) {
    return Promise.resolve(null);
  }
  state.recording = null;
  renderSessionBar();
  updateExpoButtons();
  return new Promise((resolve) => {
    const finish = () => {
      recording.stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(recording.chunks, {
        type: recording.recorder.mimeType || "audio/webm",
      });
      resolve(blob.size > 0 ? blob : null);
    };
    if (recording.recorder.state === "inactive") {
      finish();
      return;
    }
    recording.recorder.addEventListener("stop", finish, { once: true });
    recording.recorder.stop();
  });
}

// ---------------------------------------------------------------------------
// Exposition
// ---------------------------------------------------------------------------

function startSlideTimer() {
  slideStartedAt = Date.now();
  stopSlideTimer();
  slideTimer = setInterval(updateTimer, 250);
  updateTimer();
}

function stopSlideTimer() {
  if (slideTimer) {
    clearInterval(slideTimer);
  }
  slideTimer = null;
}

function currentSlideElapsed() {
  return slideStartedAt ? (Date.now() - slideStartedAt) / 1000 : 0;
}

function realElapsedSeconds() {
  return state.recordingStartMs
    ? Math.max(0, (Date.now() - state.recordingStartMs) / 1000)
    : 0;
}

function maybeWarnOverTime(total) {
  const targets = (state.session && state.session.targets) || {};
  const maxMinutes = Number(targets.max_minutes) || 18;
  if (total > maxMinutes * 60 && !state.overTimeWarned) {
    state.overTimeWarned = true;
    showNotice(
      `Superaste el tiempo objetivo (${maxMinutes} min). Puedes continuar, pero cuida el cierre.`
    );
  }
}

function updateTimer() {
  if (state.expoMode === "real") {
    const total =
      state.expoRunning && state.recordingStartMs ? realElapsedSeconds() : 0;
    $("timer").textContent = formatSeconds(total);
    maybeWarnOverTime(total);
    return;
  }
  $("timer").textContent = formatSeconds(currentSlideElapsed());
}

function renderExposition() {
  const slide = currentSlide();
  const image = $("slide-image");
  const empty = $("slide-empty");
  const counter = $("slide-counter");
  if (!slide) {
    image.hidden = true;
    empty.hidden = false;
    empty.textContent =
      "No hay láminas. Ejecuta scripts/build_context.py para generarlas en data/slides.";
    counter.textContent = "";
    return;
  }
  image.src = slide.image_url;
  image.alt = `Lámina ${slide.number}`;
  image.hidden = false;
  empty.hidden = true;
  counter.textContent = `Lámina ${slide.number} de ${state.slides.length}`;
}

function renderSlideStatuses() {
  const list = $("slide-status-list");
  list.innerHTML = "";
  state.slides.forEach((slide, index) => {
    const item = document.createElement("li");
    item.className =
      "slide-status" + (index === state.slideIndex ? " current" : "");
    const label = STATUS_LABELS[slide.transcription_status] || slide.transcription_status;
    item.innerHTML = `
      <span class="slide-number">Lámina ${escapeHtml(slide.number)}</span>
      <span class="status ${escapeHtml(slide.transcription_status)}">${escapeHtml(label)}</span>
      ${slide.duration_seconds !== null && slide.duration_seconds !== undefined ? `<span class="duration">${formatSeconds(slide.duration_seconds)}</span>` : ""}
    `;
    if (slide.transcription_error) {
      item.title = slide.transcription_error;
    }
    list.appendChild(item);
  });
}

function updateExpoButtons() {
  const session = state.session;
  const hasSlides = state.slides.length > 0;
  const finished = session && session.exposition_status === "finished";
  const running = state.expoRunning;
  const busy = state.busy;
  const real = state.expoMode === "real";
  const start = $("btn-start");
  start.disabled = !session || !hasSlides || running || finished || busy;
  if (real) {
    start.textContent =
      session && session.elapsed_seconds > 0 ? "Reanudar defensa" : "Comenzar defensa";
  } else {
    start.textContent =
      session && session.elapsed_seconds > 0 ? "Reanudar" : "Comenzar";
  }
  $("expo-mode").disabled = !session || running || busy;
  $("btn-rerecord").hidden = real;
  $("btn-rerecord").disabled = !session || !running || busy || real;
  $("btn-prev").hidden = !real;
  $("btn-prev").disabled =
    !real || !running || busy || state.slideIndex <= 0;
  $("btn-next").disabled =
    !running || busy || state.slideIndex >= state.slides.length - 1;
  $("btn-finish").disabled = !running || busy;
  const directButton = $("btn-jury-direct");
  if (directButton) {
    directButton.disabled =
      !session ||
      running ||
      busy ||
      Boolean(state.recording) ||
      session.jury.status !== "idle";
  }
  $("timer-label").textContent = real ? "Tiempo total" : "Lámina";
  $("rec-indicator").hidden = !state.recording;
}

function nextSlide() {
  if (state.expoMode === "real") {
    changeSlideReal(1);
    return;
  }
  advanceSlide();
}

function changeSlideReal(delta) {
  if (state.expoMode !== "real" || !state.expoRunning || state.busy) {
    return;
  }
  const target = state.slideIndex + delta;
  if (target < 0 || target >= state.slides.length) {
    return;
  }
  state.slideIndex = target;
  const slide = currentSlide();
  if (slide && state.recordingStartMs) {
    state.timeline.push({
      slide: slide.number,
      start_ms: Math.round(realElapsedSeconds() * 1000),
    });
  }
  renderExposition();
  renderSlideStatuses();
  updateExpoButtons();
}

function showOpening(opening) {
  state.openingUrl = (opening && opening.audio_url) || null;
  $("opening-panel").hidden = false;
  $("opening-text").textContent =
    (opening && opening.text) || "El jurado te da la palabra.";
  $("btn-play-opening").disabled = !state.openingUrl;
}

function playAudioAndWait(url, role) {
  if (!url) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const audio = new Audio(url);
    const stopSpeaking = speakCharacter(audio, role);
    let settled = false;
    const done = () => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      stopSpeaking();
      resolve();
    };
    const timer = setTimeout(done, 15000);
    audio.addEventListener("ended", done);
    audio.addEventListener("error", done);
    const attempt = audio.play();
    if (attempt && attempt.catch) {
      attempt.catch(done);
    }
  });
}

async function beginRealExposition() {
  if (!state.session || state.busy) {
    return;
  }
  state.busy = true;
  updateExpoButtons();
  try {
    const opening = await api("/api/exposition/opening", { method: "POST" });
    showOpening(opening);
    await playAudioAndWait(opening.audio_url, "juez_presidente");
    await startMic("exposition");
    state.recordingStartMs = Date.now();
    const session = await withThinking("Preparando la defensa…", () =>
      api("/api/exposition/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "real" }),
      })
    );
    state.session = session;
    state.slides = session.slides || state.slides;
    state.realStarted = true;
    state.overTimeWarned = false;
    const slide = currentSlide();
    state.timeline = slide ? [{ slide: slide.number, start_ms: 0 }] : [];
    state.expoRunning = true;
    startSlideTimer();
    renderAll();
  } catch (error) {
    await stopMic();
    state.realStarted = false;
    state.recordingStartMs = null;
    showError(error.message);
  } finally {
    state.busy = false;
    updateExpoButtons();
  }
}

async function beginExposition() {
  if (state.expoMode === "real") {
    await beginRealExposition();
    return;
  }
  if (!state.session || state.busy) {
    return;
  }
  state.busy = true;
  updateExpoButtons();
  try {
    await startMic("exposition");
    const session = await withThinking("Preparando la exposición…", () =>
      api("/api/exposition/start", { method: "POST" })
    );
    state.session = session;
    state.slides = session.slides || state.slides;
    state.expoRunning = true;
    startSlideTimer();
    renderAll();
  } catch (error) {
    await stopMic();
    showError(error.message);
  } finally {
    state.busy = false;
    updateExpoButtons();
  }
}

async function rerecordSlide() {
  if (state.busy) {
    return;
  }
  state.busy = true;
  updateExpoButtons();
  try {
    await stopMic();
    await startMic("exposition");
    startSlideTimer();
    state.expoRunning = true;
    renderAll();
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    updateExpoButtons();
  }
}

async function uploadCurrentSlide() {
  const slide = currentSlide();
  if (!slide) {
    return;
  }
  const duration = currentSlideElapsed();
  const blob = await stopMic();
  if (!blob) {
    showError(
      `No se grabó audio en la lámina ${slide.number}. Puedes regrabarla antes de continuar.`
    );
    return;
  }
  const form = new FormData();
  form.append("file", blob, audioFilename(`lamina-${slide.number}`, blob));
  form.append("duration", String(duration));
  try {
    await withThinking(`Guardando la lámina ${slide.number}…`, () =>
      api(`/api/exposition/${slide.number}/audio`, {
        method: "POST",
        body: form,
      })
    );
    slide.has_audio = true;
    slide.transcription_status = "transcribing";
    slide.transcription_error = null;
    slide.duration_seconds = duration;
    renderSlideStatuses();
    renderSessionBar();
  } catch (error) {
    showError(`No se pudo subir la lámina ${slide.number}: ${error.message}`);
  }
}

function browseSlide(delta) {
  const target = state.slideIndex + delta;
  if (target < 0 || target >= state.slides.length) {
    return;
  }
  state.slideIndex = target;
  renderExposition();
  renderSlideStatuses();
  updateExpoButtons();
}

async function goToPracticeSlide(target) {
  if (state.busy || target < 0 || target >= state.slides.length) {
    return;
  }
  state.busy = true;
  updateExpoButtons();
  try {
    await uploadCurrentSlide();
    state.slideIndex = target;
    startSlideTimer();
    renderExposition();
    renderSlideStatuses();
    try {
      await startMic("exposition");
    } catch (error) {
      showError(error.message);
    }
  } finally {
    state.busy = false;
    updateExpoButtons();
  }
}

async function advanceSlide() {
  if (!state.expoRunning || state.busy) {
    return;
  }
  if (state.slideIndex >= state.slides.length - 1) {
    return;
  }
  await goToPracticeSlide(state.slideIndex + 1);
}

async function previousSlide() {
  if (state.busy) {
    return;
  }
  if (state.slideIndex <= 0) {
    return;
  }
  if (!state.expoRunning) {
    browseSlide(-1);
    return;
  }
  // Going back must not overwrite the previous slide's transcript: save the
  // current take, pause the per-slide recording and show the earlier slide.
  state.busy = true;
  updateExpoButtons();
  try {
    await uploadCurrentSlide();
    stopSlideTimer();
    state.expoRunning = false;
    state.slideIndex -= 1;
    renderExposition();
    renderSlideStatuses();
    showNotice(
      "Lámina anterior. Pulsa Reanudar para seguir grabando o cambia de lámina con las flechas."
    );
  } finally {
    state.busy = false;
    updateExpoButtons();
  }
}

async function terminateExposition() {
  if (state.expoMode === "real") {
    await terminateRealExposition();
    return;
  }
  if (!state.expoRunning || state.busy) {
    return;
  }
  state.busy = true;
  updateExpoButtons();
  try {
    await uploadCurrentSlide();
    stopSlideTimer();
    state.expoRunning = false;
    const body = await withThinking("Guardando la práctica…", () =>
      api("/api/session/finish", { method: "POST" })
    );
    state.session = body.session;
    state.slides = body.session.slides || state.slides;
    $("expo-summary").hidden = false;
    $("expo-summary-text").textContent =
      `Práctica guardada en ${body.saved_path}. ` +
      `Duración total: ${formatSeconds(body.session.elapsed_seconds)}.`;
    renderAll();
    switchTab("jury");
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    updateExpoButtons();
  }
}

async function terminateRealExposition() {
  if (!state.expoRunning || state.busy) {
    return;
  }
  state.busy = true;
  updateExpoButtons();
  try {
    const duration = realElapsedSeconds();
    const blob = await stopMic();
    stopSlideTimer();
    state.expoRunning = false;
    state.realStarted = false;
    if (blob) {
      const form = new FormData();
      form.append("file", blob, audioFilename("defensa", blob));
      form.append("timeline", JSON.stringify(state.timeline));
      form.append("duration", String(duration));
      await withThinking("Subiendo la defensa completa…", () =>
        api("/api/exposition/full/audio", { method: "POST", body: form })
      );
      state.slides.forEach((slide) => {
        if (state.timeline.some((mark) => mark.slide === slide.number)) {
          slide.has_audio = true;
          slide.transcription_status = "transcribing";
        }
      });
      renderSlideStatuses();
      renderSessionBar();
    } else {
      showError("No se grabó audio en la defensa.");
    }
    const body = await withThinking("Guardando la práctica…", () =>
      api("/api/session/finish", { method: "POST" })
    );
    state.session = body.session;
    state.slides = body.session.slides || state.slides;
    $("expo-summary").hidden = false;
    $("expo-summary-text").textContent =
      `Defensa guardada en ${body.saved_path}. ` +
      `Duración total: ${formatSeconds(duration)}. ` +
      "La transcripción completa sigue en segundo plano; espera a que termine antes de generar la retroalimentación.";
    renderAll();
    switchTab("jury");
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    updateExpoButtons();
  }
}

function startPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
  }
  pollTimer = setInterval(async () => {
    if (!state.session) {
      return;
    }
    const jury = state.session.jury;
    const deliberating = Boolean(jury && jury.status === "deliberating");
    const pending =
      state.slides.some(
        (slide) => slide.transcription_status === "transcribing"
      ) || deliberating;
    if (!pending) {
      return;
    }
    try {
      const session = await api("/api/session");
      state.session = session;
      state.slides = session.slides || state.slides;
      renderSlideStatuses();
      renderSessionBar();
      if (session.jury && session.jury.status === "done" && state.tab === "jury") {
        renderJury();
      } else if (deliberating) {
        renderJury();
      }
    } catch (_) {
      /* keep polling silently */
    }
  }, 1600);
}

// ---------------------------------------------------------------------------
// Jury
// ---------------------------------------------------------------------------

function juryIsDeliberating(jury) {
  return Boolean(jury && jury.status === "deliberating");
}

function playJuryClosing(jury) {
  const final = jury && jury.final;
  if (!final || !final.closing_audio_url || state.juryClosingPlayed) {
    return;
  }
  state.juryClosingPlayed = true;
  playCharacterAudio(final.closing_audio_url, "juez_presidente");
}

function renderFinalVerdict(jury) {
  const final = jury.final;
  const scale = final.scale_max || 20;
  const scope = (final.chapters || []).join(", ");
  $("jury-final-scope").textContent = scope ? `Preguntas sobre: ${scope}` : "";
  $("jury-final-scope").hidden = !scope;
  const gradeLabel =
    final.final_grade !== null && final.final_grade !== undefined
      ? `${final.final_grade} / ${scale}`
      : "Sin calificación";
  $("jury-final-grade").textContent = gradeLabel;
  const extras = [];
  if (final.mention) {
    extras.push(final.mention);
  }
  if (final.no_exposition) {
    extras.push("Sin exposición (modo directo)");
  }
  $("jury-final-result").textContent =
    `${final.result || "Sin calificación"}` +
    (extras.length ? ` · ${extras.join(" · ")}` : "");
  $("jury-final-closing").textContent = final.closing || "";
  $("btn-play-closing").hidden = !final.closing_audio_url;
  $("jury-verdict-export").hidden = false;

  $("jury-final-verdicts").innerHTML = (final.verdicts || [])
    .map((item) => {
      const nota = item.nota !== null && item.nota !== undefined
        ? `${item.nota}/${scale}`
        : "Sin nota";
      const exposicion = item.exposicion !== null && item.exposicion !== undefined
        ? `${item.exposicion}/${scale}`
        : "—";
      const respuestas = item.respuestas !== null && item.respuestas !== undefined
        ? `${item.respuestas}/${scale}`
        : "—";
      return `
        <article class="verdict verdict-card">
          <div class="verdict-character">${characters.figure(
            item.role,
            characters.emotionFromScore(item.nota, scale)
          )}</div>
          <div class="verdict-body">
            <header class="verdict-head">
              <strong>${escapeHtml(item.name || item.role)}</strong>
              <span class="badge">${escapeHtml(nota)}</span>
            </header>
            <p class="metrics">Exposición: ${escapeHtml(exposicion)} · Respuestas: ${escapeHtml(respuestas)}</p>
            <div class="feedback-columns">
              <div><h4>Fortalezas</h4>${renderList(item.fortalezas)}</div>
              <div><h4>Debilidades</h4>${renderList(item.debilidades)}</div>
            </div>
            <p class="verdict-comment">${escapeHtml(item.comentario || "")}</p>
            ${
              item.parse_error
                ? '<p class="warning">No se pudo interpretar la evaluación de este juez; su nota no cuenta para el promedio.</p>'
                : ""
            }
          </div>
        </article>`;
    })
    .join("");

  const questions = final.questions || [];
  $("jury-final-questions").innerHTML = questions.length
    ? `<ol class="verdict-question-list">${questions
        .map(
          (item) => `
        <li>
          <div class="verdict-question-head">
            <strong>${escapeHtml(item.judge || item.role)}</strong>
            <span class="badge">${escapeHtml(
              TURN_KIND_LABELS[item.kind] || "Pregunta"
            )}</span>
          </div>
          <p class="q">P: ${escapeHtml(item.question)}</p>
          <p class="a">R: ${escapeHtml(item.answer)}</p>
        </li>`
        )
        .join("")}</ol>`
    : '<p class="hint">Sin preguntas registradas.</p>';

  if (state.awaitingClosing) {
    state.awaitingClosing = false;
    playJuryClosing(jury);
  }
}

function renderJury() {
  const jury = state.session ? state.session.jury : null;
  if (!jury) {
    return;
  }
  const finished = state.session.exposition_status === "finished";
  const deliberating = juryIsDeliberating(jury);
  $("jury-setup").hidden = jury.status !== "idle";
  $("btn-jury-start").disabled = !finished || jury.status !== "idle";
  $("btn-jury-direct-start").disabled =
    jury.status !== "idle" || state.busy;
  $("jury-hint").textContent = finished
    ? "El jurado está listo. Elige cuántas preguntas quieres por juez."
    : "Termina la exposición para que el jurado te haga las preguntas.";
  $("jury-deliberating").hidden = !deliberating;
  renderJuryScope(jury);
  renderJudges(jury);

  const awaiting = jury.status === "awaiting_answer" && Boolean(jury.current_question);
  $("jury-question-panel").hidden = !awaiting;
  const answerInput = $("jury-answer-text");
  if (answerInput) {
    updateAnswerControls(jury);
  }
  if (awaiting) {
    const judge = jury.judges[jury.current_judge] || {};
    const turnLabel = TURN_KIND_LABELS[jury.turn_kind] || "";
    $("jury-current-judge").textContent =
      `${judge.name || ""} · ${ROLE_LABELS[judge.role] || ""}` +
      (turnLabel ? ` · ${turnLabel}` : "");
    $("jury-question").textContent = jury.current_question;
    $("btn-play-question").disabled = !jury.current_audio_url;
  }

  const done = jury.status === "done" && Boolean(jury.final);
  $("jury-done").hidden = !done;
  if (done) {
    renderFinalVerdict(jury);
  }
}

function selectedJuryChapters() {
  const all = $("jury-chapter-all");
  if (!all || all.checked) {
    return [];
  }
  return Array.from(document.querySelectorAll(".jury-chapter:checked")).map(
    (box) => box.value
  );
}

function updateChapterControls() {
  const all = $("jury-chapter-all");
  if (!all) {
    return;
  }
  const boxes = Array.from(document.querySelectorAll(".jury-chapter"));
  if (all.checked) {
    boxes.forEach((box) => {
      box.checked = false;
      box.disabled = true;
    });
    return;
  }
  boxes.forEach((box) => {
    box.disabled = false;
  });
}

function onChapterAllChange() {
  updateChapterControls();
  if ($("jury-chapter-all").checked) {
    return;
  }
  const first = document.querySelector(".jury-chapter");
  if (first) {
    first.focus();
  }
}

function onChapterChange() {
  const all = $("jury-chapter-all");
  const boxes = Array.from(document.querySelectorAll(".jury-chapter"));
  if (all) {
    all.checked = !boxes.some((box) => box.checked);
  }
  updateChapterControls();
}

async function loadJuryChapters() {
  const field = $("jury-chapter-field");
  const options = $("jury-chapter-options");
  if (!field || !options) {
    return;
  }
  try {
    const body = await api("/api/chapters");
    const chapters = (body && body.chapters) || [];
    if (!chapters.length) {
      field.hidden = true;
      return;
    }
    options.innerHTML = chapters
      .map(
        (chapter) =>
          `<label class="checkbox"><input type="checkbox" class="jury-chapter" ` +
          `value="${escapeHtml(chapter.id)}"> ${escapeHtml(chapter.label)}</label>`
      )
      .join("");
    options.querySelectorAll(".jury-chapter").forEach((box) => {
      box.addEventListener("change", onChapterChange);
    });
    field.hidden = false;
    updateChapterControls();
  } catch (_) {
    field.hidden = true;
  }
}

function renderJuryScope(jury) {
  const element = $("jury-scope");
  if (!element) {
    return;
  }
  const scope = (jury && jury.scope_label) || "";
  element.textContent = scope ? `Preguntas sobre: ${scope}` : "";
  element.hidden = !scope;
}

function judgeByRole(jury, role) {  return (jury.judges || []).find((judge) => judge.role === role) || null;
}

function seatState(judge, jury) {
  if (!judge) {
    return { label: "", className: "" };
  }
  const scale = (jury.final && jury.final.scale_max) || 20;
  if (judge.verdict) {
    return {
      label:
        judge.score !== null && judge.score !== undefined
          ? `Veredicto · ${judge.score}/${scale}`
          : "Veredicto emitido",
      className: "done",
    };
  }
  if (jury.status === "done") {
    return { label: "Veredicto emitido", className: "done" };
  }
  if (jury.status === "awaiting_answer" && jury.current_role === judge.role) {
    return { label: "Escuchando…", className: "asking" };
  }
  return { label: "En espera", className: "" };
}

function updateAnswerControls(jury) {
  const input = $("jury-answer-text");
  if (!input) {
    return;
  }
  const canAnswer = Boolean(
    jury && jury.status === "awaiting_answer" && !state.juryBusy
  );
  input.disabled = !canAnswer;
  $("btn-jury-answer").disabled = !canAnswer;
  const ptt = $("btn-jury-ptt");
  if (ptt) {
    ptt.disabled = !canAnswer;
  }
}

/*
 * Mood of one judge right now: thinking while deliberating, the tone of the
 * question while asking, and the grade once the verdict is in.
 */
function seatEmotion(role, judge, jury, thinking) {
  if (thinking) {
    return "pensativo";
  }
  if (judge && judge.verdict) {
    const scale = (jury.final && jury.final.scale_max) || 20;
    return characters.emotionFromScore(judge.score, scale);
  }
  if (jury.status === "awaiting_answer" && jury.current_role === role) {
    return characters.emotionFromText(jury.current_question, jury.turn_kind);
  }
  return "neutral";
}

function setText(element, text) {
  if (element && element.textContent !== text) {
    element.textContent = text;
  }
}

/*
 * The seats are built once and then updated in place, so the breathing,
 * blinking and speaking animations never restart on a poll and an emotion
 * change can play its reaction.
 */
function renderJuryStage(jury) {
  const stage = $("jury-stage");
  if (!stage) {
    return;
  }
  if (!stage.querySelector(".jury-scene")) {
    const seats = JURY_LAYOUT.map(
      (role) => `
      <div class="seat${role === "juez_presidente" ? " president" : ""}" data-role="${role}">
        <div class="char-bubble" hidden>
          <span class="bubble-kind"></span>
          <p></p>
        </div>
        <div class="think-bubble" hidden>pensando…</div>
        <div class="char-figure">${characters.figure(role, "neutral")}</div>
        <div class="listen-indicator" hidden>🎙 Te escucho…</div>
        <div class="nameplate">
          <strong class="plate-name"></strong>
          <span class="plate-role">${escapeHtml(ROLE_LABELS[role] || role)}${
            role === "juez_presidente" ? " · preside" : ""
          }</span>
          <span class="seat-state"></span>
          <span class="seat-remaining"></span>
        </div>
      </div>`
    ).join("");
    stage.innerHTML = `
      <div class="jury-scene">
        <div class="jury-seats">${seats}</div>
        <div class="jury-table" aria-hidden="true"></div>
      </div>`;
  }
  const deliberating = juryIsDeliberating(jury);
  JURY_LAYOUT.forEach((role) => {
    const seat = stage.querySelector(`.seat[data-role="${role}"]`);
    if (!seat) {
      return;
    }
    const judge = judgeByRole(jury, role);
    const active =
      jury.status === "awaiting_answer" && jury.current_role === role;
    const thinking = deliberating || seat.dataset.thinking === "1";
    seat.classList.toggle("active", active);
    seat.classList.toggle("deliberating", deliberating);

    const bubble = seat.querySelector(".char-bubble");
    bubble.hidden = !active;
    setText(bubble.querySelector(".bubble-kind"), TURN_KIND_LABELS[jury.turn_kind] || "Pregunta");
    setText(bubble.querySelector("p"), active ? jury.current_question || "" : "");
    seat.querySelector(".think-bubble").hidden = !thinking;

    setText(
      seat.querySelector(".plate-name"),
      judge && judge.name ? judge.name : characters.defaultName(role)
    );
    const info = seatState(judge, jury);
    const label = seat.querySelector(".seat-state");
    label.className = `seat-state ${deliberating ? "thinking" : info.className}`;
    if (seat.dataset.activity !== "listening") {
      setText(label, deliberating ? "pensando…" : info.label);
    }
    const remaining = judge ? judge.remaining : 0;
    setText(
      seat.querySelector(".seat-remaining"),
      deliberating ? "" : `${remaining} ${remaining === 1 ? "pregunta" : "preguntas"}`
    );

    if (deliberating) {
      seat.dataset.deliberation = "1";
      characters.setActivity(seat, "thinking");
    } else if (seat.dataset.deliberation === "1") {
      delete seat.dataset.deliberation;
      if (seat.dataset.activity === "thinking") {
        characters.setActivity(seat, "idle");
      }
    }
    // While the student answers, the live reactions own the faces.
    if (!(liveReaction && jury.status === "awaiting_answer") || thinking) {
      characters.setEmotion(seat, seatEmotion(role, judge, jury, thinking));
    }
  });
}

function renderJudges(jury) {
  renderJuryStage(jury);
  const container = $("judges-list");
  const scale = (jury.final && jury.final.scale_max) || 20;
  container.innerHTML = "";
  jury.judges.forEach((judge) => {
    const card = document.createElement("article");
    const asking = judge.active && jury.status === "awaiting_answer";
    card.className = "judge-card" + (judge.active && jury.status !== "done" ? " active" : "");
    let stateLabel = "En espera";
    let stateClass = "";
    if (judge.verdict) {
      stateLabel = "Veredicto emitido";
      stateClass = "done";
    } else if (asking) {
      stateLabel = "Escuchando…";
      stateClass = "asking";
    }
    const remaining = judge.remaining === 1 ? "1 pregunta" : `${judge.remaining} preguntas`;
    const turns = judge.turns
      .map(
        (turn) => `
        <div class="turn">
          <span class="turn-kind">${escapeHtml(
            TURN_KIND_LABELS[turn.kind] || "Pregunta"
          )}</span>
          <p class="q">P: ${escapeHtml(turn.question)}</p>
          <p class="a">R: ${escapeHtml(turn.answer)}</p>
        </div>`
      )
      .join("");
    card.innerHTML = `
      <header>
        <h3>${escapeHtml(judge.name)}</h3>
        <span class="badge">${escapeHtml(ROLE_LABELS[judge.role] || judge.role)}</span>
      </header>
      <p class="persona">${escapeHtml(judge.persona)}</p>
      <p class="judge-state ${stateClass}">${escapeHtml(stateLabel)} · ${escapeHtml(remaining)}</p>
      ${
        judge.score !== null && judge.score !== undefined
          ? `<p class="score">Puntaje: ${escapeHtml(judge.score)}/${scale}</p>`
          : ""
      }
      ${judge.verdict ? `<p class="persona">${escapeHtml(judge.verdict)}</p>` : ""}
      ${turns ? `<div class="turns">${turns}</div>` : ""}
    `;
    container.appendChild(card);
  });
}

function setSeatThinking(role, active) {
  document.querySelectorAll(".seat").forEach((seat) => {
    if (seat.dataset.role !== role) {
      return;
    }
    const bubble = seat.querySelector(".think-bubble");
    if (bubble) {
      bubble.hidden = !active;
    }
    if (active) {
      seat.dataset.thinking = "1";
      characters.setActivity(seat, "thinking");
      characters.setEmotion(seat, "pensativo");
    } else {
      delete seat.dataset.thinking;
      if (seat.dataset.activity === "thinking") {
        characters.setActivity(seat, "idle");
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Live judge reactions while the student answers
// ---------------------------------------------------------------------------

// Voice above this RMS counts as speaking; faces change at most this often.
const VOICE_LEVEL = 0.025;
const REACTION_MIN_MS = 1800;
const REACTION_TICK_MS = 300;

let liveReaction = null;

function startLiveReactions() {
  if (liveReaction) {
    return;
  }
  const now = performance.now();
  liveReaction = {
    started: now,
    lastActivity: now,
    mood: null,
    analyser: null,
    source: null,
    buffer: null,
    changedAt: {},
    timer: setInterval(tickLiveReactions, REACTION_TICK_MS),
  };
}

function stopLiveReactions() {
  if (!liveReaction) {
    return;
  }
  clearInterval(liveReaction.timer);
  if (liveReaction.source) {
    try {
      liveReaction.source.disconnect();
    } catch (_) {
      // Already disconnected with the stopped stream.
    }
  }
  liveReaction = null;
}

/* Hook an analyser on the answer microphone once the recording exists. */
function attachVoiceMeter() {
  const recording = state.recording;
  if (!liveReaction || liveReaction.analyser || !recording || recording.owner !== "jury") {
    return;
  }
  const context = audioContext();
  if (!context) {
    return;
  }
  try {
    const source = context.createMediaStreamSource(recording.stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    liveReaction.source = source;
    liveReaction.analyser = analyser;
    liveReaction.buffer = new Uint8Array(analyser.fftSize);
  } catch (_) {
    // No meter: the reactions still follow time and typed words.
  }
}

function voiceLevel() {
  const { analyser, buffer } = liveReaction;
  if (!analyser) {
    return 0;
  }
  analyser.getByteTimeDomainData(buffer);
  let sum = 0;
  for (let index = 0; index < buffer.length; index += 1) {
    const value = (buffer[index] - 128) / 128;
    sum += value * value;
  }
  return Math.sqrt(sum / buffer.length);
}

function tickLiveReactions() {
  const jury = state.session && state.session.jury;
  if (!liveReaction || !jury || jury.status !== "awaiting_answer") {
    stopLiveReactions();
    return;
  }
  attachVoiceMeter();
  const now = performance.now();
  if (voiceLevel() >= VOICE_LEVEL) {
    liveReaction.lastActivity = now;
  }
  const signals = {
    elapsed: (now - liveReaction.started) / 1000,
    // Typing pauses are naturally longer than spoken ones. Without a voice
    // meter a silence cannot be measured, so it never counts against you.
    silence:
      liveReaction.typed || liveReaction.analyser
        ? (now - liveReaction.lastActivity) / 1000 / (liveReaction.typed ? 2 : 1)
        : 0,
    mood: liveReaction.mood,
  };
  document.querySelectorAll("#jury-stage .seat").forEach((seat) => {
    const role = seat.dataset.role;
    const next = characters.listeningEmotion(role, signals);
    const pose = seat.querySelector(".char-pose");
    if (!pose || pose.dataset.emotion === next) {
      return;
    }
    const last = liveReaction.changedAt[role] || 0;
    if (now - last < REACTION_MIN_MS) {
      return;
    }
    liveReaction.changedAt[role] = now;
    characters.setEmotion(seat, next);
  });
}

/* Typing an answer counts as talking; its words set the judges' mood. */
function noteTypedAnswer(text) {
  const jury = state.session && state.session.jury;
  if (!jury || jury.status !== "awaiting_answer") {
    return;
  }
  startLiveReactions();
  liveReaction.typed = true;
  liveReaction.lastActivity = performance.now();
  liveReaction.mood = characters.answerMood ? characters.answerMood(text) : null;
}

function setJuryListening(active) {
  const jury = state.session && state.session.jury;
  if (!jury || !jury.current_role) {
    return;
  }
  const seat = characterRoot(jury.current_role);
  if (!seat) {
    return;
  }
  const indicator = seat.querySelector(".listen-indicator");
  if (indicator) {
    indicator.hidden = !active;
  }
  if (active) {
    characters.setActivity(seat, "listening");
    startLiveReactions();
  } else {
    stopLiveReactions();
    if (seat.dataset.activity === "listening") {
      characters.setActivity(seat, "idle");
    }
  }
  const label = seat.querySelector(".seat-state");
  if (label) {
    label.textContent = active ? "Te escucho…" : "Escuchando…";
  }
}

function setTutorThinking(active) {
  const seat = characterRoot("tutor");
  if (!seat) {
    return;
  }
  const bubble = seat.querySelector(".think-bubble");
  if (bubble) {
    bubble.hidden = !active;
  }
  if (active) {
    const pose = seat.querySelector(".char-pose");
    const current = pose ? pose.dataset.emotion : "neutral";
    if (current !== "pensativo") {
      seat.dataset.restEmotion = current;
    }
    characters.setActivity(seat, "thinking");
    characters.setEmotion(seat, "pensativo");
  } else if (seat.dataset.activity === "thinking") {
    characters.setActivity(seat, "idle");
    characters.setEmotion(seat, seat.dataset.restEmotion || "neutral");
  }
  const label = seat.querySelector(".seat-state");
  if (label) {
    label.textContent = active ? "Pensando…" : "En línea";
  }
}

/* The tutor reacts to what it just said or to the score it just gave. */
function setTutorEmotion(emotion) {
  const seat = characterRoot("tutor");
  if (!seat) {
    return;
  }
  seat.dataset.restEmotion = emotion;
  characters.setEmotion(seat, emotion);
}

async function beginJury() {
  if (!state.session || state.busy) {
    return;
  }
  const questions = Number($("jury-questions").value) || 2;
  state.busy = true;
  $("btn-jury-start").disabled = true;
  setSeatThinking("juez_presidente", true);
  // Warm the optional filler map so the first answer can start its audio
  // immediately; a missing endpoint is ignored.
  void loadJuryFillers();
  let failure = null;
  try {
    const body = await withThinking("El jurado está pensando…", () =>
      api("/api/jury/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questions_per_judge: questions,
          chapters: selectedJuryChapters(),
        }),
      })
    );
    state.session.jury = body.jury;
    setSeatThinking("juez_presidente", false);
    renderJury();
    playCharacterAudioList(juryAudioUrls(body.jury), body.jury.current_role);
    if (body.result && body.result.notice) {
      showNotice(body.result.notice);
    }
  } catch (error) {
    failure = error.message;
    showError(error.message);
  } finally {
    state.busy = false;
    setSeatThinking("juez_presidente", false);
    if (failure) {
      renderJury();
      $("jury-hint").textContent = failure;
    }
  }
}

async function beginDirectJury() {
  if (!state.session || state.busy) {
    return;
  }
  if (state.expoRunning || state.recording) {
    showNotice("Termina la exposición antes de entrar al jurado directo.");
    return;
  }
  const jury = state.session.jury;
  if (!jury || jury.status !== "idle") {
    showNotice("El jurado ya está en marcha.");
    return;
  }
  const questions = Number($("jury-questions").value) || 2;
  state.busy = true;
  state.juryClosingPlayed = false;
  state.awaitingClosing = false;
  updateExpoButtons();
  $("btn-jury-direct-start").disabled = true;
  $("btn-jury-start").disabled = true;
  let failure = null;
  try {
    const body = await withThinking("El jurado entra en sala…", () =>
      api("/api/jury/direct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questions_per_judge: questions,
          chapters: selectedJuryChapters(),
        }),
      })
    );
    state.session.jury = body.jury;
    renderJury();
    const greeting = body.jury.greeting;
    if (greeting && greeting.audio_url) {
      await playAudioAndWait(greeting.audio_url, "juez_presidente");
    }
    playCharacterAudioList(juryAudioUrls(body.jury), body.jury.current_role);
    if (body.result && body.result.notice) {
      showNotice(body.result.notice);
    }
  } catch (error) {
    failure = error.message;
    showError(error.message);
  } finally {
    state.busy = false;
    updateExpoButtons();
    if (failure) {
      renderJury();
    }
  }
}

function openDirectJury() {
  switchTab("jury");
  beginDirectJury();
}

async function submitJuryAnswer(buildRequest) {
  if (!state.session || state.juryBusy) {
    return false;
  }
  const jury = state.session.jury;
  if (!jury || jury.status !== "awaiting_answer") {
    return false;
  }
  // Bind the submission to the question it answers (T11). A duplicate click
  // or a late push-to-talk release then becomes a stale turn on the server
  // instead of recording the same exchange twice.
  const turnId = jury.current_turn_id;
  state.juryBusy = true;
  const thinkingRole = jury.current_role;
  updateAnswerControls(jury);
  setSeatThinking(thinkingRole, true);
  // Play the asking judge's filler now, while the brain is still thinking.
  // It is optional: a missing endpoint resolves to null and nothing happens.
  const token = ++fillerToken;
  void playJudgeFiller(thinkingRole, token);
  try {
    const body = await withThinking("El juez está pensando…", () =>
      buildRequest(turnId)
    );
    fillerToken += 1;
    stopJudgeFiller();
    state.session.jury = body.jury;
    setSeatThinking(thinkingRole, false);
    renderJury();
    playCharacterAudioList(juryAudioUrls(body.jury), body.jury.current_role);
    if (body.result && body.result.notice) {
      showNotice(body.result.notice);
    }
    if (body.result && body.result.type === "deliberating") {
      state.awaitingClosing = true;
      showNotice("El jurado está deliberando…");
    }
    return true;
  } catch (error) {
    fillerToken += 1;
    stopJudgeFiller();
    showError(error.message);
    if (error.status === 409) {
      // The turn already advanced: resync so the UI shows the real question.
      try {
        await refreshSession();
      } catch (_) {
        /* the banner already explains the duplicate */
      }
    }
    return false;
  } finally {
    state.juryBusy = false;
    setSeatThinking(thinkingRole, false);
    updateAnswerControls(state.session && state.session.jury);
  }
}

async function sendJuryAnswer(blob) {
  return submitJuryAnswer((turnId) => {
    const form = new FormData();
    form.append("file", blob, audioFilename("respuesta", blob));
    if (turnId !== null && turnId !== undefined) {
      form.append("turn_id", String(turnId));
    }
    return api("/api/jury/answer/audio", { method: "POST", body: form });
  });
}

async function sendJuryText() {
  const input = $("jury-answer-text");
  if (!input || input.disabled) {
    return;
  }
  const text = input.value.trim();
  if (!text) {
    return;
  }
  stopLiveReactions();
  const sent = await submitJuryAnswer((turnId) =>
    api("/api/jury/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, turn_id: turnId }),
    })
  );
  if (sent && input) {
    input.value = "";
  }
}

// ---------------------------------------------------------------------------
// Feedback: per slide and global report
// ---------------------------------------------------------------------------

function transcriptionPending() {
  return state.slides.some(
    (slide) => slide.transcription_status === "transcribing"
  );
}

async function refreshSession() {
  const session = await api("/api/session");
  state.session = session;
  state.slides = session.slides || state.slides;
  renderSlideStatuses();
  renderSessionBar();
  renderJury();
  updateExpoButtons();
}

// Learning loop: grades over saved rehearsals, slides that keep scoring low,
// and the filler habits overall.
async function loadProgress() {
  try {
    renderProgress(await api("/api/progress"));
  } catch (error) {
    $("progress-card").hidden = true;
  }
}

function sparkline(values, max) {
  const points = values.filter((value) => typeof value === "number");
  if (points.length < 2) return "";
  const width = 160;
  const height = 36;
  const step = width / (points.length - 1);
  const coords = points
    .map((value, i) => `${(i * step).toFixed(1)},${(height - (value / max) * height).toFixed(1)}`)
    .join(" ");
  return `<svg class="sparkline" viewBox="-2 -2 ${width + 4} ${height + 4}" aria-hidden="true">
    <polyline points="${coords}" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
  </svg>`;
}

function renderProgress(progress) {
  const card = $("progress-card");
  const practices = progress.practices || [];
  if (!practices.length) {
    card.hidden = true;
    return;
  }
  const graded = practices.filter((item) => typeof item.jury_grade === "number");
  const last = graded[graded.length - 1];
  const trend = progress.jury_trend;
  const trendLabel =
    typeof trend !== "number"
      ? ""
      : `<span class="trend ${trend > 0 ? "up" : trend < 0 ? "down" : ""}">${
          trend > 0 ? "▲ +" : trend < 0 ? "▼ " : "= "
        }${escapeHtml(trend)}</span>`;
  const slideAverages = practices.map((item) => item.slide_average);
  const weak = (progress.weak_slides || [])
    .map(
      (slide) =>
        `<li><a href="#practice/${escapeHtml(slide.number)}">Lámina ${escapeHtml(slide.number)}</a>
         <span class="hint">${escapeHtml(slide.average)}/10 · ${escapeHtml(slide.attempts)} ${
          slide.attempts === 1 ? "toma" : "tomas"
        }</span></li>`
    )
    .join("");
  card.innerHTML = `
    <h2>Tu progreso</h2>
    <p class="hint">${escapeHtml(practices.length)} prácticas guardadas.</p>
    <div class="progress-grid">
      <div class="progress-stat">
        <span class="label">Última nota del jurado</span>
        <strong>${last ? `${escapeHtml(last.jury_grade)}/${escapeHtml(last.scale_max || 20)}` : "—"}</strong>
        ${trendLabel}
        ${sparkline(graded.map((item) => item.jury_grade), last ? last.scale_max || 20 : 20)}
      </div>
      <div class="progress-stat">
        <span class="label">Promedio por lámina</span>
        <strong>${
          slideAverages.filter((v) => typeof v === "number").slice(-1)[0] ?? "—"
        }<small>/10</small></strong>
        ${sparkline(slideAverages, 10)}
      </div>
      <div class="progress-stat">
        <span class="label">Repasa estas láminas</span>
        ${weak ? `<ul class="weak-slides">${weak}</ul>` : '<p class="hint">Ninguna por debajo de 7.</p>'}
      </div>
    </div>
    ${renderMuletillas(progress.muletillas)}`;
  card.hidden = false;
}

async function loadFeedback() {
  loadProgress();
  if (!state.session) {
    return;
  }
  try {
    const body = await api("/api/feedback");
    state.report = body.report;
    state.global = body.report.global || null;
    renderFeedback();
  } catch (error) {
    if (error.status !== 404) {
      showError(error.message);
    }
  }
}

async function generateFeedback() {
  if (!state.session || state.busy) {
    return;
  }
  if (transcriptionPending()) {
    showNotice(
      "Espera a que termine la transcripción de la defensa antes de evaluar."
    );
    return;
  }
  const withTranscript = state.slides.filter(
    (slide) => expositionWordCount(slide.transcript) > 0
  );
  if (!withTranscript.length) {
    showError("Primero expón al menos una lámina para poder evaluarla.");
    return;
  }
  const slides = withTranscript.filter((slide) => !slide.feedback);
  if (!slides.length) {
    await loadFeedback();
    showNotice("La retroalimentación de todas las láminas ya está generada.");
    if (!state.global) {
      await generateGlobalReport({ silent: true, force: true });
    }
    return;
  }
  const button = $("btn-feedback");
  const progress = $("feedback-progress");
  state.busy = true;
  button.disabled = true;
  progress.hidden = false;
  try {
    // Two evaluations at a time: the model slows down sharply with 3+
    // concurrent requests, so a wider pool would be slower, not faster.
    let done = 0;
    let next = 0;
    const showProgress = () => {
      progress.innerHTML = `Evaluando láminas… ${done} de ${slides.length}
        <progress max="${slides.length}" value="${done}"></progress>`;
    };
    showProgress();
    const worker = async () => {
      while (next < slides.length) {
        const slide = slides[next];
        next += 1;
        await api(`/api/feedback/${slide.number}`, { method: "POST" });
        done += 1;
        showProgress();
      }
    };
    await withThinking(`El evaluador revisa ${slides.length} láminas…`, () =>
      Promise.all([worker(), worker()])
    );
    await refreshSession();
    await loadFeedback();
    progress.textContent = "Retroalimentación lista.";
    if (!state.global) {
      await generateGlobalReport({ silent: true, force: true });
    }
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    button.disabled = false;
    setTimeout(() => {
      progress.hidden = true;
    }, 3000);
  }
}

async function generateGlobalReport(options) {
  const force = Boolean(options && options.force);
  if (!state.session || (state.busy && !force)) {
    return;
  }
  if (transcriptionPending()) {
    showNotice(
      "Espera a que termine la transcripción completa antes del informe global."
    );
    return;
  }
  const button = $("btn-global");
  const progress = $("feedback-progress");
  state.busy = true;
  button.disabled = true;
  progress.hidden = false;
  progress.textContent = "El evaluador prepara el informe global…";
  try {
    const body = await withThinking(
      "El evaluador prepara el informe global…",
      () => api("/api/feedback/global", { method: "POST" })
    );
    state.report = body.report;
    state.global = body.global;
    renderFeedback();
    progress.textContent = "Informe global listo.";
    if (!options || !options.silent) {
      showNotice("Informe global listo.");
    }
  } catch (error) {
    showError(error.message);
    progress.textContent = "";
  } finally {
    state.busy = false;
    button.disabled = false;
    setTimeout(() => {
      progress.hidden = true;
    }, 3000);
  }
}

function renderList(items) {
  if (!items || !items.length) {
    return "<ul><li>Sin observaciones.</li></ul>";
  }
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function scoreLabel(value) {
  return value === null || value === undefined ? "—" : `${value}/10`;
}

function renderGlobalFeedback(container, report) {
  const target = report.target || {};
  const estado = report.within_target
    ? "dentro del objetivo"
    : "fuera del objetivo";
  const rubric = (report.rubrica || [])
    .map(
      (item) => `
      <li><strong>${escapeHtml(item.label)}:</strong> ${escapeHtml(scoreLabel(item.puntaje))}${
        item.comentario ? ` — ${escapeHtml(item.comentario)}` : ""
      }</li>`
    )
    .join("");
  const priorities = (report.prioridades || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
  const weak = (report.laminas_debiles || [])
    .map(
      (item) =>
        `Lámina ${escapeHtml(item.number)}${
          item.puntaje !== null && item.puntaje !== undefined
            ? ` (${escapeHtml(item.puntaje)}/10)`
            : ""
        }`
    )
    .join(", ");
  const article = document.createElement("article");
  article.className = "feedback-card global-card";
  article.innerHTML = `
    <header>
      <h2>Informe global</h2>
      <span class="badge">${escapeHtml(report.duration_label || "00:00")}</span>
    </header>
    <p class="hint">
      Tiempo total: ${escapeHtml(report.duration_label || "00:00")} ·
      objetivo ${escapeHtml(target.min_label || "15:00")}–${escapeHtml(target.max_label || "18:00")} ·
      ${escapeHtml(estado)}
    </p>
    <p class="metrics">
      Palabras por minuto:
      <strong>${report.words_per_minute === null || report.words_per_minute === undefined ? "—" : escapeHtml(report.words_per_minute)}</strong>
      · Muletillas: <strong>${escapeHtml(report.muletillas || 0)}</strong>
    </p>
    ${renderMuletillas(report.muletillas_detalle)}
    ${
      report.parse_error
        ? '<p class="warning">No se pudo interpretar la respuesta del evaluador; se muestran solo las métricas calculadas.</p>'
        : ""
    }
    <div class="feedback-columns">
      <div><h4>Rúbrica CUAM</h4><ul>${rubric || "<li>Sin puntajes.</li>"}</ul></div>
      <div><h4>Prioridades</h4>${
        priorities ? `<ol>${priorities}</ol>` : '<p class="hint">Sin prioridades.</p>'
      }</div>
    </div>
    <p class="suggestion"><strong>Láminas más débiles:</strong> ${weak || "Sin láminas calificadas."}</p>
  `;
  container.appendChild(article);
}

function renderFeedback() {
  const container = $("feedback-report");
  const globalContainer = $("feedback-global");
  const exportLink = $("feedback-export");
  const report = state.report;
  container.innerHTML = "";
  globalContainer.innerHTML = "";
  if (!report) {
    exportLink.hidden = true;
    return;
  }
  const global = report.global || state.global;
  if (global) {
    renderGlobalFeedback(globalContainer, global);
  }
  exportLink.hidden = report.totals.evaluated === 0 && !global;
  report.slides.forEach((slide) => {
    const card = document.createElement("article");
    card.className = "feedback-card";
    const score =
      slide.puntaje !== null && slide.puntaje !== undefined
        ? ` · ${escapeHtml(slide.puntaje)}/10`
        : "";
    let html = `
      <header>
        <h3>Lámina ${escapeHtml(slide.number)}</h3>
        <span class="badge">${escapeHtml(slide.duration_label)}${score}</span>
      </header>`;
    if (slide.status === "no_transcript" || slide.status === "insufficient") {
      html += `<p class="hint">${escapeHtml(
        slide.message || "Sin exposición suficiente en esta lámina."
      )}</p>`;
    } else if (slide.status === "missing") {
      html += '<p class="hint">Sin retroalimentación generada.</p>';
    } else {
      if (slide.parse_error) {
        html +=
          '<p class="warning">No se pudo interpretar la respuesta del evaluador; se muestra el texto original.</p>';
      }
      if (slide.dijo) {
        html += `<p><strong>Dijo:</strong> ${escapeHtml(slide.dijo)}</p>`;
      }
      html += `<p class="metrics">Muletillas: <strong>${escapeHtml(
        slide.muletillas || 0
      )}</strong> · Palabras por minuto: <strong>${
        slide.palabras_por_minuto === null ||
        slide.palabras_por_minuto === undefined
          ? "—"
          : escapeHtml(slide.palabras_por_minuto)
      }</strong></p>`;
      html += renderMuletillas(slide.muletillas_detalle);
      html += `
        <div class="feedback-columns">
          <div><h4>Fortalezas</h4>${renderList(slide.fortalezas)}</div>
          <div><h4>Correcciones</h4>${renderList(slide.correcciones)}</div>
          <div><h4>Omisiones</h4>${renderList(slide.omisiones)}</div>
          <div><h4>Errores vs. la tesis</h4>${renderList(slide.errores_vs_tesis)}</div>
        </div>`;
      if (slide.frase_sugerida) {
        html += `<p class="suggestion"><strong>Frase sugerida:</strong> ${escapeHtml(slide.frase_sugerida)}</p>`;
      }
      if (slide.mejoras_vs_anterior) {
        html += `<p class="suggestion"><strong>Frente a la toma anterior:</strong> ${escapeHtml(slide.mejoras_vs_anterior)}</p>`;
      }
      if (slide.parse_error && slide.raw) {
        html += `<pre class="raw">${escapeHtml(slide.raw)}</pre>`;
      }
    }
    card.innerHTML = html;
    container.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Slide by slide practice (S21)
// ---------------------------------------------------------------------------

function slideTitle(slide) {
  if (slide.number === 1) {
    return "Portada";
  }
  // Skip the section kicker ("03 · MARCO METODOLÓGICO") and page numbers.
  const first = String(slide.text || "")
    .split("\n")
    .map((line) => line.trim())
    .find(
      (line) =>
        line &&
        !/^\d+(\s*\/\s*\d+)?$/.test(line) &&
        !/^\d{2}\s*·/.test(line)
    );
  const title = first || "";
  return title.length > 48 ? `${title.slice(0, 47)}…` : title;
}

function fillSlideSelect(select, includeNone) {
  if (!select) {
    return;
  }
  const signature = state.slides.map((slide) => slide.number).join(",");
  if (select.dataset.signature === signature) {
    return;
  }
  const previous = select.value;
  select.dataset.signature = signature;
  select.innerHTML = includeNone ? '<option value="">Ninguna</option>' : "";
  state.slides.forEach((slide) => {
    const option = document.createElement("option");
    option.value = String(slide.number);
    const title = slideTitle(slide);
    option.textContent = title ? `${slide.number} · ${title}` : `Lámina ${slide.number}`;
    select.appendChild(option);
  });
  if ([...select.options].some((option) => option.value === previous)) {
    select.value = previous;
  }
}

function practiceSlide() {
  return state.slides[state.practice.index] || null;
}

function practiceRecording() {
  return Boolean(state.recording && state.recording.owner === "practice");
}

function updatePracticeTimer() {
  const startedAt = state.practice.startedAt;
  const seconds = startedAt ? (Date.now() - startedAt) / 1000 : 0;
  $("practice-timer").textContent = formatSeconds(seconds);
}

function updatePracticeControls() {
  const recording = practiceRecording();
  const busy = state.practice.busy;
  const hasSlides = state.slides.length > 0;
  const record = $("btn-practice-record");
  record.textContent = recording ? "■ Terminar lámina" : "● Grabar esta lámina";
  record.classList.toggle("danger", recording);
  record.classList.toggle("primary", !recording);
  record.disabled = !state.session || !hasSlides || busy;
  $("btn-practice-prev").disabled =
    recording || busy || state.practice.index <= 0;
  $("btn-practice-next").disabled =
    recording || busy || state.practice.index >= state.slides.length - 1;
  $("practice-slide-select").disabled = recording || busy || !hasSlides;
  $("btn-practice-ask-tutor").disabled = recording || !hasSlides;
  $("practice-rec").hidden = !recording;
  $("btn-practice-replay").hidden = !state.practice.audioUrl;
}

function setPracticeStatus(message) {
  $("practice-status").textContent = message;
}

function renderPracticeFeedback(slide) {
  const container = $("practice-feedback");
  const feedback = slide && slide.feedback;
  if (!feedback) {
    container.innerHTML = "";
    return;
  }
  const score =
    feedback.puntaje !== null && feedback.puntaje !== undefined
      ? `<span class="score-big">${escapeHtml(feedback.puntaje)}/10</span>`
      : "";
  const wpm =
    feedback.palabras_por_minuto === null || feedback.palabras_por_minuto === undefined
      ? "—"
      : escapeHtml(feedback.palabras_por_minuto);
  const duration =
    slide.duration_seconds !== null && slide.duration_seconds !== undefined
      ? formatSeconds(slide.duration_seconds)
      : "—";
  const bad = [
    ...(feedback.correcciones || []),
    ...(feedback.errores_vs_tesis || []).map((item) => `Contradice la tesis: ${item}`),
  ];
  const section = (kind, title, items) =>
    items && items.length
      ? `<section class="${kind}"><h3>${title}</h3>${renderList(items)}</section>`
      : "";
  let html = `
    <div class="score-line">
      ${score}
      <span class="badge">Duración ${duration}</span>
      <span class="badge">${wpm} palabras por minuto</span>
      <span class="badge">${escapeHtml(feedback.muletillas || 0)} muletillas</span>
    </div>
    ${renderMuletillas(feedback.muletillas_detalle)}`;
  if (feedback.parse_error) {
    html +=
      '<p class="warning">No se pudo interpretar la respuesta del tutor; se muestra el texto original.</p>';
    if (feedback.raw) {
      html += `<pre class="raw">${escapeHtml(feedback.raw)}</pre>`;
    }
  }
  if (feedback.comentario_tutor) {
    html += `<p class="tutor-says">“${escapeHtml(feedback.comentario_tutor)}”</p>`;
  }
  if (feedback.mejoras_vs_anterior) {
    html += `<section class="progress-note"><h3>Frente a tu toma anterior</h3><p>${escapeHtml(
      feedback.mejoras_vs_anterior
    )}</p></section>`;
  }
  html += section("good", "✔ Lo que dijiste bien", feedback.fortalezas);
  html += section("bad", "✘ Lo que dijiste mal", bad);
  html += section("missing", "Te faltó mencionar", feedback.omisiones);
  if (feedback.frase_sugerida) {
    html += `<section class="tip"><h3>Frase que puedes usar</h3><p>${escapeHtml(
      feedback.frase_sugerida
    )}</p></section>`;
  }
  if (slide.transcript) {
    html += `<details><summary>Lo que dijiste (transcripción)</summary><p>${escapeHtml(
      slide.transcript
    )}</p></details>`;
  }
  container.innerHTML = html;
}

function renderPracticeHistory(slide) {
  const history = $("practice-history");
  const local = (slide && state.practice.attempts[slide.number]) || [];
  // The server keeps every evaluated take, so the history survives a reload.
  const saved = slide
    ? [
        ...(slide.attempts || []).map((attempt) => attempt.puntaje),
        ...(slide.feedback ? [slide.feedback.puntaje] : []),
      ]
    : [];
  const attempts = saved.length >= local.length ? saved : local;
  if (!attempts.length) {
    history.hidden = true;
    return;
  }
  history.hidden = false;
  history.textContent = `Intentos en esta lámina: ${attempts
    .map((score) => (score === null || score === undefined ? "—" : `${score}/10`))
    .join(" → ")}`;
}

function renderPractice() {
  const select = $("practice-slide-select");
  if (!select) {
    return;
  }
  fillSlideSelect(select, false);
  if (state.practice.index >= state.slides.length) {
    state.practice.index = 0;
  }
  const slide = practiceSlide();
  const image = $("practice-slide-image");
  const empty = $("practice-slide-empty");
  if (!slide) {
    image.hidden = true;
    empty.hidden = false;
  } else {
    if (image.dataset.src !== slide.image_url) {
      image.src = slide.image_url;
      image.dataset.src = slide.image_url;
    }
    image.alt = `Lámina ${slide.number}`;
    image.hidden = false;
    empty.hidden = true;
    select.value = String(slide.number);
  }
  if (!state.practice.busy && !practiceRecording()) {
    setPracticeStatus(
      slide && slide.feedback
        ? "Esta es la última retroalimentación de esta lámina. Grábala de nuevo para mejorarla."
        : "Graba la lámina para recibir la retroalimentación."
    );
  }
  renderPracticeFeedback(slide);
  renderPracticeHistory(slide);
  updatePracticeControls();
}

function goToPracticeIndex(index) {
  if (
    practiceRecording() ||
    state.practice.busy ||
    index < 0 ||
    index >= state.slides.length
  ) {
    return;
  }
  state.practice.index = index;
  state.practice.audioUrl = null;
  renderPractice();
}

async function waitForTranscription(number) {
  const deadline = Date.now() + 180000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 400));
    const status = await api(`/api/exposition/${number}/status`);
    if (status.transcription_status !== "transcribing") {
      // One full refresh once the transcript is ready.
      const session = await api("/api/session");
      state.session = session;
      state.slides = session.slides || state.slides;
      return state.slides.find((item) => item.number === number) || status;
    }
  }
  throw new Error("La transcripción tardó demasiado. Intenta grabar de nuevo.");
}

async function startPracticeRecording() {
  try {
    await startMic("practice");
  } catch (error) {
    showError(error.message);
    return;
  }
  state.practice.audioUrl = null;
  state.practice.startedAt = Date.now();
  clearInterval(state.practice.timer);
  state.practice.timer = setInterval(updatePracticeTimer, 250);
  updatePracticeTimer();
  setPracticeStatus("Grabando… explica la lámina y pulsa Terminar cuando acabes.");
  updatePracticeControls();
}

async function finishPracticeRecording() {
  const slide = practiceSlide();
  const duration = state.practice.startedAt
    ? (Date.now() - state.practice.startedAt) / 1000
    : 0;
  clearInterval(state.practice.timer);
  state.practice.timer = null;
  state.practice.startedAt = null;
  state.practice.busy = true;
  const blob = await stopMic();
  updatePracticeControls();
  try {
    if (!blob || !slide) {
      throw new Error("No se grabó audio. Intenta de nuevo.");
    }
    const number = slide.number;
    const form = new FormData();
    form.append("file", blob, audioFilename(`lamina-${number}`, blob));
    form.append("duration", String(Math.round(duration * 100) / 100));
    setPracticeStatus("Transcribiendo lo que dijiste…");
    $("practice-feedback").innerHTML = "";
    await api(`/api/exposition/${number}/audio`, { method: "POST", body: form });
    const transcribed = await waitForTranscription(number);
    if (transcribed.transcription_status !== "done") {
      throw new Error(
        transcribed.transcription_error ||
          "No se entendió el audio. Revisa el micrófono y graba de nuevo."
      );
    }
    setPracticeStatus("El tutor está revisando tu explicación…");
    setTutorThinking(true);
    const body = await api(`/api/feedback/${number}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speak: $("practice-speak").checked }),
    });
    const target = state.slides.find((item) => item.number === number);
    if (target) {
      target.feedback = body.feedback;
    }
    const attempts = state.practice.attempts[number] || [];
    attempts.push(body.feedback ? body.feedback.puntaje : null);
    state.practice.attempts[number] = attempts;
    state.practice.audioUrl = body.audio_url || null;
    state.practice.busy = false;
    renderPractice();
    setTutorThinking(false);
    setTutorEmotion(
      characters.emotionFromScore(body.feedback ? body.feedback.puntaje : null, 10)
    );
    setPracticeStatus("Listo. Lee la retroalimentación o graba la lámina otra vez.");
    if (body.audio_url) {
      playCharacterAudio(body.audio_url, "tutor");
    }
  } catch (error) {
    setPracticeStatus("No se pudo evaluar esta toma.");
    showError(error.message);
  } finally {
    setTutorThinking(false);
    state.practice.busy = false;
    updatePracticeControls();
    renderSessionBar();
  }
}

function togglePracticeRecording() {
  if (state.practice.busy) {
    return;
  }
  if (practiceRecording()) {
    finishPracticeRecording();
  } else {
    startPracticeRecording();
  }
}

function askTutorAboutPracticeSlide() {
  const slide = practiceSlide();
  if (!slide) {
    return;
  }
  switchTab("tutor");
  const select = $("tutor-slide-select");
  fillSlideSelect(select, true);
  select.value = String(slide.number);
  const input = $("tutor-input");
  if (!input.value.trim()) {
    input.value = slide.feedback
      ? "¿Cómo puedo explicar mejor esta lámina?"
      : "¿Qué debo decir en esta lámina?";
  }
  input.focus();
}

// ---------------------------------------------------------------------------
// Tutor
// ---------------------------------------------------------------------------

function renderTutorCharacter() {
  const container = $("tutor-character");
  if (!container) {
    return;
  }
  const tutor = (state.session && state.session.tutor) || {};
  const name = tutor.name || characters.defaultName("tutor");
  if (container.dataset.ready === "1" && container.dataset.name === name) {
    return;
  }
  container.dataset.ready = "1";
  container.dataset.name = name;
  container.innerHTML = `
    <div class="seat tutor-seat" data-role="tutor">
      <div class="think-bubble" hidden>pensando…</div>
      <div class="char-figure">${characters.figure("tutor")}</div>
      <div class="nameplate">
        <strong class="plate-name">${escapeHtml(name)}</strong>
        <span class="plate-role">Tutor académico</span>
        <span class="seat-state">En línea</span>
      </div>
    </div>`;
}

function renderTutorAttachments() {
  const container = $("tutor-attachments");
  if (!container) {
    return;
  }
  container.innerHTML = "";
  container.hidden = state.tutorImages.length === 0;
  state.tutorImages.forEach((image, index) => {
    const thumb = document.createElement("div");
    thumb.className = "attachment-thumb";
    thumb.innerHTML = `<img src="${escapeHtml(image.url)}" alt="Imagen ${index + 1}">
      <span class="thumb-label">${escapeHtml(image.name)}</span>
      <button class="ghost" type="button" aria-label="Quitar imagen">×</button>`;
    thumb.querySelector("button").addEventListener("click", () => {
      state.tutorImages.splice(index, 1);
      renderTutorAttachments();
    });
    container.appendChild(thumb);
  });
}

const TUTOR_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
const TUTOR_MAX_IMAGES = 4;
const TUTOR_MAX_SIDE = 1600;

function readImageFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("No se pudo leer la imagen."));
    reader.onload = () => {
      const original = String(reader.result || "");
      const image = new Image();
      image.onerror = () => reject(new Error("El archivo no es una imagen válida."));
      image.onload = () => {
        const side = Math.max(image.naturalWidth, image.naturalHeight);
        if (side <= TUTOR_MAX_SIDE && file.size <= 2 * 1024 * 1024) {
          resolve(original);
          return;
        }
        // Large screenshots or photos are scaled down so they travel fast.
        const scale = Math.min(1, TUTOR_MAX_SIDE / side);
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(image.naturalWidth * scale);
        canvas.height = Math.round(image.naturalHeight * scale);
        canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.88));
      };
      image.src = original;
    };
    reader.readAsDataURL(file);
  });
}

async function addTutorImages(files) {
  for (const file of files) {
    if (!TUTOR_IMAGE_TYPES.includes(file.type)) {
      showError("Solo puedes adjuntar imágenes PNG, JPG, WEBP o GIF.");
      continue;
    }
    if (state.tutorImages.length >= TUTOR_MAX_IMAGES) {
      showError(`Puedes adjuntar hasta ${TUTOR_MAX_IMAGES} imágenes por mensaje.`);
      break;
    }
    try {
      const url = await readImageFile(file);
      state.tutorImages.push({ url, name: file.name || "imagen pegada" });
    } catch (error) {
      showError(error.message);
    }
  }
  renderTutorAttachments();
}

function takeTutorAttachments() {
  const select = $("tutor-slide-select");
  const slide = select && select.value ? Number(select.value) : null;
  const images = state.tutorImages.map((image) => image.url);
  return { slide, images };
}

function clearTutorAttachments() {
  state.tutorImages = [];
  const select = $("tutor-slide-select");
  if (select) {
    select.value = "";
  }
  renderTutorAttachments();
}

function slideImageUrl(number) {
  const slide = state.slides.find((item) => item.number === number);
  return slide ? slide.image_url : null;
}

function renderTutor() {
  renderTutorCharacter();
  fillSlideSelect($("tutor-slide-select"), true);
  const container = $("tutor-messages");
  const messages =
    state.session && state.session.tutor ? state.session.tutor.messages : [];
  container.innerHTML = "";
  if (!messages.length) {
    container.innerHTML =
      '<p class="hint">Aún no hay mensajes. Pregunta lo que necesites.</p>';
    return;
  }
  messages.forEach((message) => {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${message.role === "user" ? "user" : "tutor"}`;
    const shown = [];
    if (message.slide) {
      const url = slideImageUrl(message.slide);
      if (url) {
        shown.push({ url, label: `Lámina ${message.slide}` });
      }
    }
    (message.images || []).forEach((url, index) => {
      shown.push({ url, label: `Imagen ${index + 1}` });
    });
    if (shown.length) {
      const gallery = document.createElement("div");
      gallery.className = "bubble-images";
      shown.forEach((item) => {
        const link = document.createElement("a");
        link.href = item.url;
        link.target = "_blank";
        link.rel = "noopener";
        link.innerHTML = `<img src="${escapeHtml(item.url)}" alt="${escapeHtml(
          item.label
        )}"><span class="thumb-label">${escapeHtml(item.label)}</span>`;
        gallery.appendChild(link);
      });
      bubble.appendChild(gallery);
    }
    if (message.text) {
      const text = document.createElement("div");
      text.textContent = message.text;
      bubble.appendChild(text);
    }
    container.appendChild(bubble);
  });
  container.scrollTop = container.scrollHeight;
}

async function postTutor(path, options) {
  if (state.tutorBusy) {
    return;
  }
  state.tutorBusy = true;
  setTutorThinking(true);
  try {
    const body = await withThinking("El tutor está pensando…", () =>
      api(path, options)
    );
    state.session.tutor = {
      ...(state.session.tutor || {}),
      messages: body.messages,
    };
    renderTutor();
    const replies = (body.messages || []).filter((message) => message.role !== "user");
    const last = replies[replies.length - 1];
    setTutorThinking(false);
    if (last) {
      setTutorEmotion(characters.emotionFromText(last.text || last.content || ""));
    }
    if (body.audio_url) {
      playCharacterAudio(body.audio_url, "tutor");
    }
  } catch (error) {
    showError(error.message);
  } finally {
    state.tutorBusy = false;
    setTutorThinking(false);
  }
}

async function sendTutorText() {
  if (state.tutorBusy) {
    return;
  }
  const input = $("tutor-input");
  const text = input.value.trim();
  const { slide, images } = takeTutorAttachments();
  if (!text && !images.length && slide === null) {
    return;
  }
  input.value = "";
  clearTutorAttachments();
  await postTutor("/api/tutor/message", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      speak: $("tutor-speak").checked,
      slide,
      images,
    }),
  });
}

async function sendTutorAudio(blob) {
  const { slide, images } = takeTutorAttachments();
  const form = new FormData();
  form.append("file", blob, audioFilename("pregunta", blob));
  form.append("speak", String($("tutor-speak").checked));
  if (slide !== null) {
    form.append("slide", String(slide));
  }
  form.append("images", JSON.stringify(images));
  clearTutorAttachments();
  await postTutor("/api/tutor/audio", { method: "POST", body: form });
}

// ---------------------------------------------------------------------------
// Push-to-talk
// ---------------------------------------------------------------------------

function isTypingTarget(target) {
  if (!target) {
    return false;
  }
  const tag = (target.tagName || "").toLowerCase();
  return (
    tag === "input" ||
    tag === "textarea" ||
    tag === "select" ||
    target.isContentEditable === true
  );
}

function bindPushToTalk(button, isActive, onBlob, hooks) {
  const extras = hooks || {};
  let holding = false;
  let finishing = false;

  const start = async (event) => {
    if (event) {
      event.preventDefault();
    }
    if (holding || finishing || button.disabled || !isActive()) {
      return;
    }
    holding = true;
    state.pttHeld = true;
    button.classList.add("recording");
    if (extras.onStart) {
      extras.onStart();
    }
    try {
      await startMic(button.dataset.owner || "exposition");
      if (!holding) {
        await stopMic();
      }
    } catch (error) {
      holding = false;
      state.pttHeld = false;
      button.classList.remove("recording");
      if (extras.onStop) {
        extras.onStop();
      }
      showError(error.message);
    }
  };

  const stop = async (event) => {
    if (event) {
      event.preventDefault();
    }
    if (!holding) {
      return;
    }
    holding = false;
    // Keep the control locked until the upload finishes so a second keyup or
    // mouseup, or a fast new press, cannot start a second submission (T11).
    finishing = true;
    state.pttHeld = false;
    button.classList.remove("recording");
    if (extras.onStop) {
      extras.onStop();
    }
    try {
      const blob = await stopMic();
      if (blob) {
        await onBlob(blob);
      }
    } finally {
      finishing = false;
    }
  };

  button.addEventListener("mousedown", start);
  button.addEventListener("touchstart", start, { passive: false });
  window.addEventListener("mouseup", stop);
  window.addEventListener("touchend", stop);
  document.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || event.repeat || !isActive()) {
      return;
    }
    if (isTypingTarget(event.target)) {
      return;
    }
    if (event.target && event.target.blur) {
      event.target.blur();
    }
    event.preventDefault();
    start(event);
  });
  document.addEventListener("keyup", (event) => {
    if (event.code !== "Space" || !isActive()) {
      return;
    }
    event.preventDefault();
    stop(event);
  });
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function bind() {
  document.querySelectorAll(".tab-button[data-tab]").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  $("btn-new-session").addEventListener("click", newSession);
  $("btn-start").addEventListener("click", beginExposition);
  $("btn-rerecord").addEventListener("click", rerecordSlide);
  $("btn-prev").addEventListener("click", () => changeSlideReal(-1));
  $("btn-next").addEventListener("click", nextSlide);
  $("btn-finish").addEventListener("click", terminateExposition);
  $("btn-jury-direct").addEventListener("click", openDirectJury);
  $("btn-nav-direct").addEventListener("click", openDirectJury);
  $("btn-jury-direct-start").addEventListener("click", beginDirectJury);
  $("jury-chapter-all").addEventListener("change", onChapterAllChange);
  $("expo-mode").addEventListener("change", (event) => {
    if (state.expoRunning) {
      event.target.value = state.expoMode;
      showNotice("Termina la grabación actual antes de cambiar de modo.");
      return;
    }
    state.expoMode = event.target.value === "real" ? "real" : "practice";
    renderAll();
  });
  $("btn-play-opening").addEventListener("click", () =>
    playCharacterAudio(state.openingUrl, "juez_presidente")
  );
  $("btn-go-jury").addEventListener("click", () => switchTab("jury"));
  $("btn-jury-start").addEventListener("click", beginJury);
  const answerInput = $("jury-answer-text");
  if (answerInput) {
    $("btn-jury-answer").addEventListener("click", sendJuryText);
    answerInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.repeat) {
        event.preventDefault();
        sendJuryText();
      }
    });
    answerInput.addEventListener("input", () => noteTypedAnswer(answerInput.value));
  }
  $("btn-play-question").addEventListener("click", () => {
    const jury = state.session && state.session.jury;
    if (jury) {
      playCharacterAudioList(juryAudioUrls(jury), jury.current_role);
    }
  });
  $("btn-play-closing").addEventListener("click", () => {
    const jury = state.session && state.session.jury;
    if (jury && jury.final) {
      state.juryClosingPlayed = false;
      playJuryClosing(jury);
    }
  });
  $("btn-feedback").addEventListener("click", generateFeedback);
  $("btn-global").addEventListener("click", () => generateGlobalReport());
  $("btn-tutor-send").addEventListener("click", sendTutorText);
  $("btn-tutor-attach").addEventListener("click", () => $("tutor-attach-input").click());
  $("tutor-attach-input").addEventListener("change", (event) => {
    const files = [...(event.target.files || [])];
    event.target.value = "";
    addTutorImages(files);
  });
  $("tutor-input").addEventListener("paste", (event) => {
    const items = [...((event.clipboardData && event.clipboardData.items) || [])];
    const files = items
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter(Boolean);
    if (files.length) {
      event.preventDefault();
      addTutorImages(files);
    }
  });
  $("btn-practice-record").addEventListener("click", togglePracticeRecording);
  $("btn-practice-prev").addEventListener("click", () =>
    goToPracticeIndex(state.practice.index - 1)
  );
  $("btn-practice-next").addEventListener("click", () =>
    goToPracticeIndex(state.practice.index + 1)
  );
  $("practice-slide-select").addEventListener("change", (event) => {
    const index = state.slides.findIndex(
      (slide) => String(slide.number) === event.target.value
    );
    goToPracticeIndex(index);
  });
  $("btn-practice-replay").addEventListener("click", () =>
    playCharacterAudio(state.practice.audioUrl, "tutor")
  );
  $("btn-practice-ask-tutor").addEventListener("click", askTutorAboutPracticeSlide);
  $("tutor-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendTutorText();
    }
  });

  // Arrow keys change slides in every exposition mode. In practice mode they
  // behave like the on-screen slide buttons; outside a recording they only
  // browse. Space is reserved for push-to-talk in the jury and tutor tabs.
  document.addEventListener("keydown", (event) => {
    if (state.tab === "practice" && !isTypingTarget(event.target)) {
      if (event.code === "ArrowRight" || event.code === "PageDown") {
        event.preventDefault();
        goToPracticeIndex(state.practice.index + 1);
      } else if (event.code === "ArrowLeft" || event.code === "PageUp") {
        event.preventDefault();
        goToPracticeIndex(state.practice.index - 1);
      }
      return;
    }
    if (
      state.tab !== "exposition" ||
      isTypingTarget(event.target) ||
      state.pttHeld ||
      !state.slides.length
    ) {
      return;
    }
    const forward = event.code === "ArrowRight" || event.code === "PageDown";
    const backward = event.code === "ArrowLeft" || event.code === "PageUp";
    if (!forward && !backward) {
      return;
    }
    event.preventDefault();
    if (state.expoMode === "real") {
      if (state.expoRunning) {
        changeSlideReal(forward ? 1 : -1);
      } else {
        browseSlide(forward ? 1 : -1);
      }
      return;
    }
    if (!state.expoRunning) {
      browseSlide(forward ? 1 : -1);
      return;
    }
    if (forward) {
      advanceSlide();
    } else {
      previousSlide();
    }
  });

  const juryPtt = $("btn-jury-ptt");
  juryPtt.dataset.owner = "jury";
  bindPushToTalk(
    juryPtt,
    () =>
      state.tab === "jury" &&
      !state.juryBusy &&
      state.session &&
      state.session.jury.status === "awaiting_answer",
    sendJuryAnswer,
    {
      onStart: () => setJuryListening(true),
      onStop: () => setJuryListening(false),
    }
  );

  const tutorPtt = $("btn-tutor-ptt");
  tutorPtt.dataset.owner = "tutor";
  bindPushToTalk(
    tutorPtt,
    () => state.tab === "tutor" && !state.tutorBusy,
    sendTutorAudio
  );

  // Create/resume the Web Audio context on the first real gesture so the
  // judge mouths can follow the TTS level later, when the sound is not tied
  // to a direct user event.
  document.addEventListener("pointerdown", unlockAudio, { once: true });
  document.addEventListener("keydown", unlockAudio, { once: true });
}

async function init() {
  bind();
  if (characters.preload) {
    characters.preload();
  }
  void loadJuryChapters();
  try {
    await loadInitialSession();
  } catch (error) {
    showError(error.message);
  }
  openTabFromHash();
  startPolling();
}

// "#practice/13" opens the slide-by-slide tab on slide 13; "#tutor" etc. too.
function openTabFromHash() {
  const [name, number] = window.location.hash.replace(/^#/, "").split("/");
  if (!name || !document.getElementById(`tab-${name}`)) {
    return;
  }
  if (name === "practice" && number) {
    const index = state.slides.findIndex((slide) => String(slide.number) === number);
    if (index >= 0) {
      state.practice.index = index;
    }
  }
  switchTab(name);
}

document.addEventListener("DOMContentLoaded", init);
window.addEventListener("hashchange", openTabFromHash);
