# Character artwork

The files in this directory are static SVGs generated **offline** for this
application. The app serves them from `/static/characters/`; it never calls a
CDN and never runs a build step at runtime.

- Generator: [DiceBear](https://www.dicebear.com) `@dicebear/core` 9.4.x +
  `@dicebear/collection` 9.4.x, MIT licensed.
- Avatar style: [Avataaars](https://avataaars.com/) by
  [Pablo Stanley](https://twitter.com/pablostanley), "free for personal and
  commercial use". DiceBear embeds the same attribution in each SVG's
  `<metadata>`.
- Variations: each role has one folder per emotion
  (neutral, feliz, enojado, esceptico, sorprendido, pensativo, preocupado), and each folder has `closed.svg`,
  `half.svg`, `open.svg` (mouth states used by the speaking animation) and
  `blink.svg` (eyes closed).

Regenerate with `scripts/gen_characters.mjs`; dependencies are installed in a
temporary directory outside this repository (see the script header).

Roles: juez_metodologo (Dra. Yajaira Contreras), juez_tecnico (Ing. Jesús Rondón), juez_presidente (MSc. Carmen Rojas), tutor (Prof. Andrés Salas).
