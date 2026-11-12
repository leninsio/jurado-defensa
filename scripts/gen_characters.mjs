#!/usr/bin/env node
/*
 * Generates the jury/tutor avatars as static SVG files, offline.
 *
 * The app never talks to a CDN and never runs a build step: this script is
 * executed once by a developer and its output is committed under
 * web/characters/. It uses the DiceBear library (MIT) with the Avataaars
 * style by Pablo Stanley ("free for personal and commercial use"); see the
 * generated web/characters/LICENSE.md.
 *
 * Dependencies live in a throwaway directory OUTSIDE this repository:
 *
 *   tmp="$(mktemp -d)"
 *   cp scripts/gen_characters.mjs "$tmp/"
 *   (cd "$tmp" && pnpm add @dicebear/core@9 @dicebear/collection@9)
 *   (cd "$tmp" && node gen_characters.mjs /absolute/path/to/web/characters)
 *   rm -rf "$tmp"
 *
 * Running `node scripts/gen_characters.mjs` from the repo itself fails unless
 * @dicebear/core and @dicebear/collection are already resolvable, which is the
 * whole reason the install happens in the temp directory.
 *
 * Each role gets one folder per emotion (neutral, feliz, enojado, esceptico,
 * sorprendido, pensativo, preocupado). An emotion fixes the eyebrows and eyes;
 * inside it there are four deterministic frames: closed/half/open mouth and a
 * blink frame (eyes closed, mouth closed). web/characters.js stacks the four
 * frames and toggles their opacity, so the mouth follows the Web Audio level
 * without reloading, and swaps the emotion folder when the mood changes.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { createAvatar } from "@dicebear/core";
import { avataaars } from "@dicebear/collection";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUT = resolve(HERE, "..", "web", "characters");
const OUT_DIR = resolve(process.argv[2] || DEFAULT_OUT);

const COMMON = {
  size: 264,
  backgroundColor: ["transparent"],
  style: ["default"],
  topProbability: 100,
  accessoriesProbability: 0,
  facialHairProbability: 0,
};

/*
 * One fixed, distinct look per role, matching config.json:
 * - Yajaira (~45): dark hair up in a bun, formal charcoal blazer, glasses.
 * - Jesús (~30): short black hair, light beard, casual crew-neck shirt.
 * - Carmen (~60): silver bob, slate blazer over a sweater, no glasses.
 * - Andrés (~50): wavy brown hair, round glasses, grey cardigan, friendly.
 * Skin tones are four different natural values.
 */
const ROLES = [
  {
    role: "juez_metodologo",
    label: "Dra. Yajaira Contreras",
    seed: "yajaira-contreras-metodologa",
    options: {
      ...COMMON,
      top: ["bun"],
      hairColor: ["2c1b18"],
      skinColor: ["d08b5b"],
      clothing: ["blazerAndShirt"],
      clothesColor: ["262e33"],
      accessories: ["prescription02"],
      accessoriesColor: ["3c4f5c"],
      accessoriesProbability: 100,
      eyebrows: ["default"],
      eyes: ["default"],
    },
    mouthClosed: "serious",
  },
  {
    role: "juez_tecnico",
    label: "Ing. Jesús Rondón",
    seed: "jesus-rondon-tecnico",
    options: {
      ...COMMON,
      top: ["shortFlat"],
      hairColor: ["2c1b18"],
      skinColor: ["ae5d29"],
      clothing: ["shirtCrewNeck"],
      clothesColor: ["25557c"],
      facialHair: ["beardLight"],
      facialHairColor: ["2c1b18"],
      facialHairProbability: 100,
      eyebrows: ["default"],
      eyes: ["default"],
    },
    mouthClosed: "default",
  },
  {
    role: "juez_presidente",
    label: "MSc. Carmen Rojas",
    seed: "carmen-rojas-presidenta",
    options: {
      ...COMMON,
      top: ["bob"],
      hairColor: ["e8e1e1"],
      skinColor: ["edb98a"],
      clothing: ["blazerAndSweater"],
      clothesColor: ["3c4f5c"],
      eyebrows: ["raisedExcited"],
      eyes: ["default"],
    },
    mouthClosed: "default",
  },
  {
    role: "tutor",
    label: "Prof. Andrés Salas",
    seed: "andres-salas-tutor",
    options: {
      ...COMMON,
      top: ["shortWaved"],
      hairColor: ["724133"],
      skinColor: ["614335"],
      clothing: ["collarAndSweater"],
      clothesColor: ["929598"],
      accessories: ["round"],
      accessoriesColor: ["262e33"],
      accessoriesProbability: 100,
      eyebrows: ["raisedExcited"],
      eyes: ["default"],
    },
    mouthClosed: "twinkle",
  },
];

/*
 * Facial expression per emotion. `closed: null` keeps the role's own resting
 * mouth; half/open are the speaking shapes, chosen so the mouth still reads as
 * the emotion while it moves (a smile stays a smile, a frown stays a frown).
 */
export const EMOTIONS = {
  neutral: { eyebrows: null, eyes: "default", closed: null, half: "disbelief", open: "screamOpen" },
  feliz: { eyebrows: "raisedExcitedNatural", eyes: "happy", closed: "twinkle", half: "smile", open: "smile" },
  enojado: { eyebrows: "angry", eyes: "squint", closed: "grimace", half: "grimace", open: "screamOpen" },
  esceptico: { eyebrows: "upDown", eyes: "default", closed: "serious", half: "disbelief", open: "screamOpen" },
  sorprendido: { eyebrows: "raisedExcited", eyes: "surprised", closed: "disbelief", half: "disbelief", open: "screamOpen" },
  pensativo: { eyebrows: "flatNatural", eyes: "side", closed: "serious", half: "disbelief", open: "screamOpen" },
  preocupado: { eyebrows: "sadConcernedNatural", eyes: "default", closed: "sad", half: "concerned", open: "concerned" },
};

function svgFor(role, emotion, mouth, eyes) {
  const eyebrows = emotion.eyebrows || role.options.eyebrows[0];
  return createAvatar(avataaars, {
    seed: role.seed,
    ...role.options,
    eyebrows: [eyebrows],
    mouth: [mouth],
    eyes: [eyes],
  }).toString();
}

function licenseNote() {
  return `# Character artwork

The files in this directory are static SVGs generated **offline** for this
application. The app serves them from \`/static/characters/\`; it never calls a
CDN and never runs a build step at runtime.

- Generator: [DiceBear](https://www.dicebear.com) \`@dicebear/core\` 9.4.x +
  \`@dicebear/collection\` 9.4.x, MIT licensed.
- Avatar style: [Avataaars](https://avataaars.com/) by
  [Pablo Stanley](https://twitter.com/pablostanley), "free for personal and
  commercial use". DiceBear embeds the same attribution in each SVG's
  \`<metadata>\`.
- Variations: each role has one folder per emotion
  (${Object.keys(EMOTIONS).join(", ")}), and each folder has \`closed.svg\`,
  \`half.svg\`, \`open.svg\` (mouth states used by the speaking animation) and
  \`blink.svg\` (eyes closed).

Regenerate with \`scripts/gen_characters.mjs\`; dependencies are installed in a
temporary directory outside this repository (see the script header).

Roles: ${ROLES.map((role) => `${role.role} (${role.label})`).join(", ")}.
`;
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  let total = 0;
  for (const role of ROLES) {
    for (const [name, emotion] of Object.entries(EMOTIONS)) {
      const dir = join(OUT_DIR, role.role, name);
      await mkdir(dir, { recursive: true });
      const closed = emotion.closed || role.mouthClosed;
      const mouths = { closed, half: emotion.half, open: emotion.open };
      for (const [state, mouth] of Object.entries(mouths)) {
        await writeFile(join(dir, `${state}.svg`), svgFor(role, emotion, mouth, emotion.eyes), "utf8");
      }
      await writeFile(join(dir, "blink.svg"), svgFor(role, emotion, closed, "closed"), "utf8");
      total += 4;
    }
    process.stdout.write(`${role.role}: ${Object.keys(EMOTIONS).length} emotions\n`);
  }
  await writeFile(join(OUT_DIR, "LICENSE.md"), licenseNote(), "utf8");
  process.stdout.write(`wrote ${total} SVGs to ${OUT_DIR}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
