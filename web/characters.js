"use strict";

/*
 * Illustrated characters for the jury table and the tutor.
 *
 * The artwork is a set of static SVGs generated offline by
 * scripts/gen_characters.mjs (DiceBear Avataaars; see web/characters/LICENSE.md)
 * and served from /static/characters/<role>/<emotion>/<frame>.svg. There is no
 * CDN, no external image host and no build step at runtime.
 *
 * Every emotion has four frames: closed / half / open mouth and one blink
 * frame (eyes closed). `figure(role, emotion)` stacks them in one portrait and
 * CSS shows only the active mouth state, which lets `setMouth` follow the real
 * Web Audio level at animation speed without reloading images. `setEmotion`
 * swaps the four frames to another emotion folder (preloaded, so the swap is
 * instant) and plays a short reaction animation. The seat wrapper gets the
 * `speaking`, `listening` and `thinking` classes from `setActivity`.
 */
(function () {
  const BASE = "/static/characters";

  const LOOKS = {
    juez_metodologo: {
      name: "Dra. Yajaira Contreras",
      aria: "Dra. Yajaira Contreras, metodóloga del jurado",
    },
    juez_tecnico: {
      name: "Ing. Jesús Rondón",
      aria: "Ing. Jesús Rondón, ingeniero de software del jurado",
    },
    juez_presidente: {
      name: "MSc. Carmen Rojas",
      aria: "MSc. Carmen Rojas, presidenta del jurado",
    },
    tutor: {
      name: "Prof. Andrés Salas",
      aria: "Prof. Andrés Salas, tutor académico",
    },
  };

  const EMOTIONS = {
    neutral: { label: "atento" },
    feliz: { label: "contento" },
    enojado: { label: "molesto" },
    esceptico: { label: "escéptico" },
    sorprendido: { label: "sorprendido" },
    pensativo: { label: "pensativo" },
    preocupado: { label: "preocupado" },
  };

  // Small symbols floating next to the head so the mood reads at any size.
  const EMOTES = {
    feliz: `<svg viewBox="0 0 24 24"><path d="M12 2l2.1 6.2L20 10l-5.9 1.8L12 18l-2.1-6.2L4 10l5.9-1.8z" fill="#ffd166"/><circle cx="19" cy="18" r="2" fill="#ffd166"/></svg>`,
    enojado: `<svg viewBox="0 0 24 24"><g fill="none" stroke="#ff4d4d" stroke-width="3" stroke-linecap="round"><path d="M4 9c3 0 5-2 5-5"/><path d="M15 4c0 3 2 5 5 5"/><path d="M20 15c-3 0-5 2-5 5"/><path d="M9 20c0-3-2-5-5-5"/></g></svg>`,
    esceptico: `<svg viewBox="0 0 24 24"><path d="M8.5 8.5a3.5 3.5 0 1 1 5.2 3.1c-1.1.6-1.7 1.3-1.7 2.6v.8" fill="none" stroke="#9fb3ff" stroke-width="2.6" stroke-linecap="round"/><circle cx="12" cy="19.5" r="1.6" fill="#9fb3ff"/></svg>`,
    sorprendido: `<svg viewBox="0 0 24 24"><path d="M12 3v11" stroke="#ffc857" stroke-width="3.2" stroke-linecap="round"/><circle cx="12" cy="19.5" r="2" fill="#ffc857"/></svg>`,
    pensativo: `<svg viewBox="0 0 24 24"><circle cx="5" cy="13" r="2.2" fill="#c8d2e8"/><circle cx="12" cy="13" r="2.2" fill="#c8d2e8"/><circle cx="19" cy="13" r="2.2" fill="#c8d2e8"/></svg>`,
    preocupado: `<svg viewBox="0 0 24 24"><path d="M12 3c3 4.5 5 7.3 5 10a5 5 0 0 1-10 0c0-2.7 2-5.5 5-10z" fill="#7cc4ff"/><path d="M10 13.5a2 2 0 0 0 2 2" fill="none" stroke="#fff" stroke-width="1.4" stroke-linecap="round" opacity="0.8"/></svg>`,
  };

  const FRAMES = [
    { state: "closed", className: "char-mouth-closed" },
    { state: "half", className: "char-mouth-half" },
    { state: "open", className: "char-mouth-open" },
    { state: "blink", className: "char-blink-frame" },
  ];

  function roleKey(role) {
    return LOOKS[role] ? role : "tutor";
  }

  function emotionKey(emotion) {
    return EMOTIONS[emotion] ? emotion : "neutral";
  }

  function frameUrl(role, emotion, state) {
    return `${BASE}/${roleKey(role)}/${emotionKey(emotion)}/${state}.svg`;
  }

  function figure(role, emotion) {
    const key = roleKey(role);
    const mood = emotionKey(emotion);
    const look = LOOKS[key];
    const layers = FRAMES.map(
      (frame) => `
        <img
          class="char-layer ${frame.className}"
          data-frame="${frame.state}"
          src="${frameUrl(key, mood, frame.state)}"
          alt=""
          aria-hidden="true"
          draggable="false"
        >`
    ).join("");
    return `
      <div class="char-pose" data-emotion="${mood}">
        <div class="char-portrait" data-role="${key}" data-mouth="closed" role="img" aria-label="${look.aria}, ${EMOTIONS[mood].label}">
          ${layers}
        </div>
        <span class="char-emote" aria-hidden="true">${EMOTES[mood] || ""}</span>
      </div>`;
  }

  function portraitOf(root) {
    if (!root) {
      return null;
    }
    if (root.classList && root.classList.contains("char-portrait")) {
      return root;
    }
    return root.querySelector(".char-portrait");
  }

  function poseOf(root) {
    if (!root) {
      return null;
    }
    if (root.classList && root.classList.contains("char-pose")) {
      return root;
    }
    return root.querySelector(".char-pose");
  }

  /*
   * Quantize the normalized audio level (0..1) into the three drawn mouth
   * shapes. DOM writes only happen when the band changes, so a 60 fps audio
   * loop does not thrash the layout.
   */
  function mouthState(level) {
    const clean = Math.max(0, Math.min(1, Number(level) || 0));
    if (clean < 0.12) {
      return "closed";
    }
    if (clean < 0.48) {
      return "half";
    }
    return "open";
  }

  function setMouth(root, level) {
    const portrait = portraitOf(root);
    if (!portrait) {
      return;
    }
    const next = mouthState(level);
    if (portrait.dataset.mouth !== next) {
      portrait.dataset.mouth = next;
    }
  }

  /*
   * Change the face to another emotion. The frames are swapped in place (the
   * mouth keeps following the audio) and a one-shot reaction plays: a shake
   * for anger, a hop for joy or surprise, a tilt for doubt.
   */
  function setEmotion(root, emotion) {
    const pose = poseOf(root);
    const portrait = portraitOf(root);
    if (!pose || !portrait) {
      return;
    }
    const mood = emotionKey(emotion);
    if (pose.dataset.emotion === mood) {
      return;
    }
    const role = portrait.dataset.role;
    pose.dataset.emotion = mood;
    portrait.querySelectorAll(".char-layer").forEach((layer) => {
      layer.src = frameUrl(role, mood, layer.dataset.frame);
    });
    const look = LOOKS[roleKey(role)];
    portrait.setAttribute("aria-label", `${look.aria}, ${EMOTIONS[mood].label}`);
    const emote = pose.querySelector(".char-emote");
    if (emote) {
      emote.innerHTML = EMOTES[mood] || "";
    }
    react(pose);
  }

  function react(pose) {
    pose.classList.remove("emote-change");
    // Force a reflow so the one-shot animation restarts on every change.
    void pose.offsetWidth;
    pose.classList.add("emote-change");
  }

  function setActivity(root, activity) {
    if (!root) {
      return;
    }
    const name = activity || "idle";
    root.classList.toggle("speaking", name === "speaking");
    root.classList.toggle("listening", name === "listening");
    root.classList.toggle("thinking", name === "thinking");
    root.dataset.activity = name;
  }

  /* Warm the browser cache with every frame so emotion swaps never flicker. */
  const preloaded = [];
  function preload(roles) {
    (roles || Object.keys(LOOKS)).forEach((role) => {
      Object.keys(EMOTIONS).forEach((mood) => {
        FRAMES.forEach((frame) => {
          const image = new Image();
          image.decoding = "async";
          image.src = frameUrl(role, mood, frame.state);
          preloaded.push(image);
        });
      });
    });
  }

  function normalize(text) {
    return String(text || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "");
  }

  const CUES = [
    [
      "enojado",
      /no (me )?respond|no contest|contradic|no coincide|incorrect|equivoc|inaceptable|no tiene sentido|no es (asi|cierto|correcto)|le repito|otra vez le|grave/,
    ],
    [
      "preocupado",
      /me preocupa|preocupante|me temo|insuficiente|debil|no queda claro|confus|le falto|faltaria/,
    ],
    [
      "feliz",
      /excelente|muy bien|felicit|buen trabajo|bien explicad|me gusta|bien hecho|enhorabuena|bienvenid|buenas (tardes|dias|noches)/,
    ],
    ["esceptico", /esta segur|realmente|demuestre|con que evidencia|como sustenta|como justific|de verdad/],
    ["sorprendido", /interesante|sorprend|curioso|no esperaba|vaya/],
  ];

  /*
   * Read the mood of a spoken line. The turn kind sets the default: a
   * follow-up is skeptical, an interruption surprised; text cues win.
   */
  function emotionFromText(text, kind) {
    const clean = normalize(text);
    for (const [mood, pattern] of CUES) {
      if (pattern.test(clean)) {
        return mood;
      }
    }
    if (kind === "repregunta") {
      return "esceptico";
    }
    if (kind === "intervencion") {
      return "sorprendido";
    }
    return "neutral";
  }

  /* Mood that matches a grade: pleased, neutral, worried or upset. */
  function emotionFromScore(score, max) {
    const value = Number(score);
    const top = Number(max) || 10;
    if (score === null || score === undefined || Number.isNaN(value)) {
      return "neutral";
    }
    const ratio = value / top;
    if (ratio >= 0.8) {
      return "feliz";
    }
    if (ratio >= 0.6) {
      return "neutral";
    }
    if (ratio >= 0.5) {
      return "preocupado";
    }
    return "enojado";
  }

  /*
   * How each judge reacts while the student answers. `attentive` rotates
   * while the student talks normally; the other keys answer a signal.
   */
  const TEMPERAMENTS = {
    juez_metodologo: {
      attentive: ["neutral", "esceptico", "neutral", "pensativo"],
      pause: "esceptico",
      longPause: "enojado",
      tooLong: "enojado",
      strong: "neutral",
      weak: "enojado",
    },
    juez_tecnico: {
      attentive: ["pensativo", "neutral", "sorprendido", "neutral"],
      pause: "pensativo",
      longPause: "esceptico",
      tooLong: "esceptico",
      strong: "sorprendido",
      weak: "preocupado",
    },
    juez_presidente: {
      attentive: ["neutral", "feliz", "neutral", "pensativo"],
      pause: "neutral",
      longPause: "preocupado",
      tooLong: "preocupado",
      strong: "feliz",
      weak: "preocupado",
    },
  };

  // Seconds: a pause, a long silence, and an answer that runs too long.
  const PAUSE_SECONDS = 2.5;
  const LONG_PAUSE_SECONDS = 6;
  const TOO_LONG_SECONDS = 90;
  // The attentive face changes every few seconds, staggered per judge.
  const ATTENTIVE_STEP_SECONDS = 5;

  const WEAK_ANSWER = /\bno (lo )?se\b|no recuerdo|no estoy segur|no me acuerdo|tal vez|quizas|mas o menos|creo que|supongo/g;
  const STRONG_ANSWER = /segun (la|mi) tesis|los resultados|la muestra|la poblacion|el objetivo|por lo tanto|en conclusion|ya que|porque|\d+ ?(%|por ciento)|cronbach|kuder|instrumento|validez|confiabilidad/g;

  /* Tone of what the student has typed so far: "weak", "strong" or null. */
  function answerMood(text) {
    const clean = normalize(text);
    if (clean.trim().split(/\s+/).length < 3) {
      return null;
    }
    const weak = (clean.match(WEAK_ANSWER) || []).length;
    const strong = (clean.match(STRONG_ANSWER) || []).length;
    if (weak > strong) {
      return "weak";
    }
    if (strong > 0) {
      return "strong";
    }
    return null;
  }

  /*
   * Emotion of one judge while the student answers.
   * signals: { elapsed, silence } in seconds and an optional `mood` from
   * answerMood. Long silences weigh most, then the tone of the words, then a
   * short pause, then length; otherwise the judge cycles attentive faces.
   */
  function listeningEmotion(role, signals) {
    const temperament = TEMPERAMENTS[role];
    if (!temperament) {
      return "neutral";
    }
    const elapsed = Math.max(0, Number(signals && signals.elapsed) || 0);
    const silence = Math.max(0, Number(signals && signals.silence) || 0);
    const mood = signals && signals.mood;
    if (silence >= LONG_PAUSE_SECONDS) {
      return temperament.longPause;
    }
    if (mood === "weak") {
      return temperament.weak;
    }
    if (silence >= PAUSE_SECONDS) {
      return temperament.pause;
    }
    if (elapsed >= TOO_LONG_SECONDS) {
      return temperament.tooLong;
    }
    if (mood === "strong") {
      return temperament.strong;
    }
    const offset = Object.keys(TEMPERAMENTS).indexOf(role);
    const step = Math.floor(elapsed / ATTENTIVE_STEP_SECONDS) + offset;
    return temperament.attentive[step % temperament.attentive.length];
  }

  function defaultName(role) {
    return (LOOKS[role] || LOOKS.tutor).name;
  }

  const api = {
    figure,
    setMouth,
    setActivity,
    setEmotion,
    emotionFromText,
    emotionFromScore,
    answerMood,
    listeningEmotion,
    preload,
    defaultName,
    roles: ["juez_metodologo", "juez_presidente", "juez_tecnico", "tutor"],
    emotions: Object.keys(EMOTIONS),
  };
  if (typeof window !== "undefined") {
    window.JuryCharacters = api;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})();
