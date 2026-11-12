# Simulador de jurado para tu defensa de tesis

Ensaya la defensa de tu trabajo de grado frente a un **jurado de inteligencia
artificial que te escucha y te pregunta por voz**. Expones lámina por lámina,
tres jueces con personalidades distintas te hacen preguntas sobre lo que
dijiste, recibes correcciones por cada lámina y puedes consultarle tus dudas a
un tutor. Todo corre en tu computadora: tu tesis y tus grabaciones no se suben
a ningún sitio web.

Fue creado para la defensa de TSU en Informática del CUAM, pero puedes
adaptarlo a tu carrera e institución cambiando un archivo de configuración.

---

## ¿Qué puedes hacer?

| Modo | Para qué sirve |
| --- | --- |
| **1 · Exposición** | Presentas tus láminas en voz alta. En **modo práctica** grabas lámina por lámina y puedes repetir cualquiera; en **defensa real** haces una sola grabación continua, con reloj, como el día de la defensa. |
| **Lámina a lámina** | Practicas una sola lámina: la grabas, el tutor la evalúa (también mira la imagen de la lámina) y te dice en voz alta qué dijiste bien y qué te faltó. Si la vuelves a grabar, te dice qué mejoraste respecto de la toma anterior. |
| **2 · Jurado** | Tres jueces te preguntan sobre **lo que expusiste**: la metodóloga (exigente con el método), el ingeniero (decisiones técnicas) y la presidenta (utilidad del trabajo y conclusiones). Respondes hablando (mantén pulsada la barra espaciadora o el botón) o escribiendo. Pueden repreguntar o interrumpir, y al final deliberan y te dan una nota. |
| **Jurado directo** | Si no quieres exponer, vas directo a las preguntas. Puedes limitarlas a ciertos capítulos. |
| **3 · Retroalimentación** | Informe por lámina: qué dijiste, fortalezas, correcciones, omisiones, errores frente a la tesis, una frase sugerida, tus **muletillas** («o sea ×3», «digamos») y tus **palabras por minuto**. También hay un informe global con rúbrica y tiempo total, y la tarjeta **Tu progreso**, que compara tus prácticas y te dice qué láminas repasar. |
| **4 · Tutor** | Chat por texto o voz para resolver dudas sobre tu tesis o las normas. Puedes mostrarle una lámina o pegarle una imagen. |

Los jueces y el tutor son personajes animados: mueven la boca al hablar y
cambian de emoción (atentos, contentos, molestos, escépticos, sorprendidos,
pensativos o preocupados) según la pregunta, tus pausas, lo que escribes y la
nota que te ponen.

---

## Lo que necesitas

- **Una Mac** (la app usa herramientas de macOS para el audio). En Windows o
  Linux puede funcionar con ajustes, pero no está probado.
- **Python 3.12**.
- **[OpenCode](https://opencode.ai)** instalado y con sesión iniciada en un
  proveedor de modelos. Por defecto, todos los personajes usan
  `opencode-go/deepseek-v4.1-flash`; si usas otro proveedor, cambia el campo
  `model` de cada rol en `config.json`.
- **Homebrew** para instalar `ffmpeg` (convierte tu voz grabada) y `poppler`
  (lee los PDF y convierte tus láminas en imágenes):

  ```bash
  brew install ffmpeg poppler
  ```

- **Un modelo de Whisper** para transcribir tu voz. Descarga
  `ggml-large-v3-turbo-q5_0.bin` desde los modelos de whisper.cpp en Hugging
  Face y guarda su ruta para el paso 4 de la instalación.
- **Chrome, Edge o Safari**, con permiso para usar el micrófono.

---

## Instalación paso a paso

1. **Descarga el proyecto**

   ```bash
   git clone https://github.com/leninsio/jurado-defensa.git
   cd jurado-defensa
   ```

2. **Crea el entorno de Python e instala las dependencias**

   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

3. **Descarga las voces** (voces neuronales en español que funcionan sin
   internet una vez descargadas)

   ```bash
   scripts/download_voices.sh
   ```

4. **Indica dónde está el modelo de Whisper** (si no lo guardaste en la ruta
   por defecto)

   ```bash
   export JURADO_WHISPER_MODEL="/ruta/a/ggml-large-v3-turbo-q5_0.bin"
   ```

---

## Adáptalo a tu tesis

Todo se configura en **`config.json`**:

1. **Tus documentos** (`sources`). Guárdalos, por ejemplo, en `~/tesis/` y
   revisa las rutas:
   - `thesis`: tu tesis en Word (`.docx`).
   - `norms`: las normas o rúbricas de evaluación de tu institución en PDF.
   - `deck`: tu presentación de la defensa **exportada a PDF**.
   - `study_guide`: cualquier guía o apunte en HTML que uses para estudiar.
2. **Tu nombre** (`student.name`): los jueces te saludan con él.
3. **El tiempo de exposición** (`jury.target_minutes_min` y
   `jury.target_minutes_max`): por defecto, de 15 a 18 minutos.
4. **Los jueces** (`roles`): nombres, personalidades y voces. Si tu carrera o
   nivel es distinto (licenciatura, ingeniería, maestría), ajusta también las
   reglas comunes en `rules`, que hoy describen una defensa de TSU.
5. **Prepara el contexto.** Este paso lee tu tesis, las normas y las láminas,
   y crea la carpeta `data/`. Repítelo cada vez que cambies la tesis o la
   presentación:

   ```bash
   .venv/bin/python scripts/build_context.py
   ```

---

## Úsalo

```bash
.venv/bin/python -m app.main
```

Abre **http://127.0.0.1:8000** en el navegador y permite el micrófono. La
primera vez puede tardar un poco mientras arrancan el modelo y las voces.

Para usar otro puerto: `JURADO_PORT=8001 .venv/bin/python -m app.main`.

**Atajos de teclado:**

- `→` y `←` cambian de lámina mientras expones.
- Mantener la **barra espaciadora** pulsada graba tu respuesta al jurado.

---

## Consejos para aprovecharlo

- **Empieza por «Lámina a lámina»** con las láminas que más te cuestan, y
  repítelas hasta que el tutor deje de marcarte omisiones.
- **Haz al menos una «defensa real» completa** para medir tu tiempo total.
- **Revisa «Tu progreso»** después de cada sesión: te muestra si tu nota del
  jurado sube y qué láminas siguen por debajo de 7.
- **Fíjate en tus muletillas**: la app te dice cuáles repites («o sea»,
  «este», «digamos») para que trabajes en ellas.
- **El jurado no te pide memorizar cifras.** Te pregunta el porqué y el cómo
  de lo que expusiste: practica explicarlo con tus propias palabras.
- **Responde en voz alta** aunque puedas escribir: así practicas como el día
  de la defensa.

---

## Privacidad

- Tu tesis, tus grabaciones y tus prácticas se guardan solo en `data/`, en tu
  computadora. Esa carpeta está excluida de git, así que no se publica aunque
  subas tu copia del proyecto.
- La transcripción de tu voz (Whisper) y las voces (Piper) funcionan sin
  conexión. Las preguntas y evaluaciones las genera el modelo de lenguaje que
  configures en OpenCode, así que el texto de tu tesis y tus respuestas sí se
  envían a ese proveedor.

---

## Problemas comunes

| Problema | Qué hacer |
| --- | --- |
| No transcribe lo que digo | Revisa que `ffmpeg` esté instalado, que el navegador tenga permiso de micrófono y que `JURADO_WHISPER_MODEL` apunte al modelo. |
| Los jueces no responden o tardan mucho | Comprueba que OpenCode tenga sesión iniciada y que el modelo de `config.json` exista en tu proveedor. Si haces varias cosas a la vez, el modelo se vuelve más lento. |
| No se escuchan las voces | Ejecuta `scripts/download_voices.sh`. Los jueces intentan primero las voces XTTS (opcionales, de mayor calidad) y, si no están instaladas, usan Piper. |
| No aparecen mis láminas | Exporta la presentación a PDF, revisa la ruta en `config.json` → `sources.deck` y vuelve a ejecutar `scripts/build_context.py`. |
| Cambié la tesis y el jurado no lo sabe | Vuelve a ejecutar `scripts/build_context.py` y reinicia la app. |

---

## Para quienes quieran contribuir

- Los tests se ejecutan con `.venv/bin/python -m pytest -q`.
- La documentación técnica completa (arquitectura, API, latencia y voces) está
  en el [README en inglés](README.md).
- Si mejoras algo, abre un *pull request* para que otros estudiantes también lo
  aprovechen.

## Licencia

MIT: puedes usarlo, modificarlo y compartirlo libremente. Los personajes se
generan con DiceBear (estilo Avataaars de Pablo Stanley); consulta
`web/characters/LICENSE.md`.
