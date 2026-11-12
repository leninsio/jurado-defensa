#!/usr/bin/env python3
"""Regenerate the TTS listening-comparison samples for the jury simulator.

The comparison keeps the app's active voices untouched: it synthesizes the
same two lines per jury role with every candidate engine and writes the wavs
plus a static ``index.html`` under ``data/voice-samples/`` (gitignored, it is
generated audio).

Engines live outside the app virtualenv, each one in its own venv under
``~/Library/Application Support/JuradoDefensa/engines/<engine>/``.  This
script orchestrates them by re-invoking itself with the engine's interpreter
(``--worker``), so the heavy models load once per engine and the reported
generation time is the synthesis call itself, not interpreter startup.

Usage
-----
    .venv/bin/python scripts/voice_samples.py                 # every engine
    .venv/bin/python scripts/voice_samples.py --engine kokoro
    .venv/bin/python scripts/voice_samples.py --list-voices   # XTTS speakers
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "data" / "voice-samples"
ENGINES_ROOT = Path(
    os.environ.get(
        "JURADO_ENGINES",
        Path.home() / "Library" / "Application Support" / "JuradoDefensa" / "engines",
    )
)
PIPER_VOICES_DIR = Path(
    os.environ.get(
        "JURADO_PIPER_VOICES",
        Path.home() / "Library" / "Application Support" / "JuradoDefensa" / "voices",
    )
)

ROLES = ["juez_metodologo", "juez_tecnico", "juez_presidente", "tutor"]
ROLE_LABELS = {
    "juez_metodologo": "Dra. Yajaira Contreras (metodóloga)",
    "juez_tecnico": "Ing. Jesús Rondón (técnico)",
    "juez_presidente": "MSc. Carmen Rojas (presidenta)",
    "tutor": "Prof. Andrés Salas (tutor)",
}
LINES = {
    1: "Buenas tardes. Adelante, Tu Nombre, ya puedes empezar.",
    2: (
        "Mencionaste que el sistema separa los permisos del cajero y del "
        "gerente. ¿Por qué era importante hacerlo en una empresa tan pequeña?"
    ),
}

# Kokoro v1.0 Spanish voices: only one female (ef_dora) and two male
# (em_alex, em_santa) ship with the model.  The presidenta uses a blend of
# ef_dora with an English female style vector, which keeps Spanish phonemes
# (espeak-ng, lang="es") but shifts the timbre; it must be auditioned because
# accent cannot be verified without listening.
KOKORO_ROLES = {
    "juez_metodologo": "ef_dora",
    "juez_tecnico": "em_alex",
    "juez_presidente": "blend:ef_dora:0.8:af_heart",
    "tutor": "em_santa",
}
KOKORO_LABELS = {
    "ef_dora": "ef_dora",
    "em_alex": "em_alex",
    "em_santa": "em_santa",
    "blend:ef_dora:0.8:af_heart": "ef_dora + af_heart (80/20 blend)",
}

# XTTS-v2 built-in studio speakers (CPML, non-commercial: fine for this
# personal rehearsal app).  Names are the model's own speaker ids.  The
# character mapping was chosen from an objective probe (median F0 and speaking
# rate over the Spanish opening line) because the mapping must be confirmed by
# ear on the comparison page: Claribel Dervla 153 Hz/5.7 s (formal 45),
# Narelle Moon 194 Hz/7.2 s (measured 60), Royston Min 128 Hz/4.1 s (casual
# 30), Gilberto Mathias 124 Hz/6.0 s (friendly 50, Spanish-sounding name).
XTTS_ROLES = {
    "juez_metodologo": "Claribel Dervla",
    "juez_tecnico": "Royston Min",
    "juez_presidente": "Narelle Moon",
    "tutor": "Gilberto Mathias",
}

ENGINE_LABELS = {
    "piper": "Piper (baseline)",
    "kokoro": "Kokoro-82M",
    "xtts": "XTTS-v2 (coqui-tts)",
}


def _venv_python(engine: str) -> Path:
    return ENGINES_ROOT / engine / "venv" / "bin" / "python"


KOKORO_MODELS_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)


def ensure_kokoro_models() -> None:
    """Download the kokoro-onnx v1.0 model and voice pack on first use."""
    models_dir = ENGINES_ROOT / "kokoro" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        target = models_dir / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        print(f"   downloading {name} (first run)…", flush=True)
        urllib.request.urlretrieve(f"{KOKORO_MODELS_URL}/{name}", target)


def piper_roles() -> dict[str, dict]:
    """Read the baseline voice per role from config.json (single source)."""
    config = json.loads((REPO / "config.json").read_text(encoding="utf-8"))
    roles: dict[str, dict] = {}
    for role in ROLES:
        voice = config["roles"][role].get("voice", "")
        if not voice.startswith("piper:"):
            raise SystemExit(f"config.json role {role} is not a piper voice: {voice!r}")
        model, _, speaker = voice[len("piper:"):].partition("#")
        roles[role] = {
            "voice": model,
            "speaker": int(speaker) if speaker else None,
            "label": f"{model}#{speaker}" if speaker else model,
        }
    return roles


def engine_spec(engine: str) -> dict[str, dict]:
    if engine == "piper":
        return piper_roles()
    if engine == "kokoro":
        return {
            role: {"voice": voice, "label": KOKORO_LABELS[voice]}
            for role, voice in KOKORO_ROLES.items()
        }
    if engine == "xtts":
        return {
            role: {"voice": voice, "label": voice}
            for role, voice in XTTS_ROLES.items()
        }
    raise SystemExit(f"unknown engine: {engine}")


# --------------------------------------------------------------------------
# workers: run inside the engine venv, load the model once, synthesize all
# --------------------------------------------------------------------------


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    data = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((data * 32767).astype(np.int16).tobytes())


def _worker_piper(spec: dict) -> dict:
    from piper import PiperVoice, SynthesisConfig

    models: dict[str, object] = {}
    items = []
    for role, settings in spec["roles"].items():
        model = settings["voice"]
        voice = models.get(model)
        if voice is None:
            model_path = PIPER_VOICES_DIR / f"{model}.onnx"
            if not model_path.is_file():
                raise SystemExit(f"Piper voice not found: {model_path}")
            voice = PiperVoice.load(str(model_path))
            models[model] = voice
        for number, text in spec["lines"].items():
            path = Path(spec["outdir"]) / f"{role}-{number}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with wave.open(str(path), "wb") as wav_file:
                voice.synthesize_wav(
                    text,
                    wav_file,
                    syn_config=SynthesisConfig(speaker_id=settings.get("speaker")),
                )
            gen_s = time.perf_counter() - started
            with wave.open(str(path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                audio_s = wav_file.getnframes() / sample_rate
            items.append(
                {
                    "role": role,
                    "n": int(number),
                    "path": str(path.relative_to(spec["root"])),
                    "voice": model,
                    "voice_label": settings.get("label") or model,
                    "gen_s": round(gen_s, 3),
                    "audio_s": round(audio_s, 3),
                    "rtf": round(gen_s / audio_s, 3) if audio_s else None,
                    "sample_rate": sample_rate,
                }
            )
    return {"engine": "piper", "items": items}


def _kokoro_voice(kokoro, name: str):
    if not name.startswith("blend:"):
        return kokoro.voices[name]
    _, first, weight, second = name.split(":")
    weight = float(weight)
    return (1.0 - weight) * kokoro.voices[first] + weight * kokoro.voices[second]


def _worker_kokoro(spec: dict) -> dict:
    from kokoro_onnx import Kokoro

    models_dir = ENGINES_ROOT / "kokoro" / "models"
    kokoro = Kokoro(
        str(models_dir / "kokoro-v1.0.onnx"),
        str(models_dir / "voices-v1.0.bin"),
    )
    items = []
    for role, settings in spec["roles"].items():
        voice = _kokoro_voice(kokoro, settings["voice"])
        for number, text in spec["lines"].items():
            path = Path(spec["outdir"]) / f"{role}-{number}.wav"
            started = time.perf_counter()
            samples, sample_rate = kokoro.create(
                text, voice=voice, speed=1.0, lang="es"
            )
            gen_s = time.perf_counter() - started
            _write_wav(path, samples, sample_rate)
            audio_s = len(samples) / sample_rate
            items.append(
                {
                    "role": role,
                    "n": int(number),
                    "path": str(path.relative_to(spec["root"])),
                    "voice": settings["voice"],
                    "voice_label": settings.get("label") or settings["voice"],
                    "gen_s": round(gen_s, 3),
                    "audio_s": round(audio_s, 3),
                    "rtf": round(gen_s / audio_s, 3) if audio_s else None,
                    "sample_rate": sample_rate,
                }
            )
    return {"engine": "kokoro", "items": items}


def _load_xtts(device: str):
    os.environ.setdefault("COQUI_TOS_AGREED", "1")  # CPML non-commercial use
    base = ENGINES_ROOT / "xtts"
    os.environ.setdefault("TTS_HOME", str(base / "tts_home"))
    os.environ.setdefault("HF_HOME", str(base / "hf_home"))
    from TTS.api import TTS

    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def _worker_xtts(spec: dict) -> dict:
    import numpy as np
    import torch

    requested = spec.get("device", "auto")
    if requested == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = requested
    tts = _load_xtts(device)
    speakers = getattr(tts, "speakers", None) or []
    missing = [s["voice"] for s in spec["roles"].values() if s["voice"] not in speakers]
    if missing:
        raise SystemExit(
            f"XTTS speakers not found: {missing}; available: {speakers}"
        )
    items = []
    for role, settings in spec["roles"].items():
        for number, text in spec["lines"].items():
            path = Path(spec["outdir"]) / f"{role}-{number}.wav"
            started = time.perf_counter()
            samples = tts.tts(text=text, speaker=settings["voice"], language="es")
            gen_s = time.perf_counter() - started
            sample_rate = tts.synthesizer.output_sample_rate
            _write_wav(path, np.asarray(samples, dtype=np.float32), sample_rate)
            audio_s = len(samples) / sample_rate
            items.append(
                {
                    "role": role,
                    "n": int(number),
                    "path": str(path.relative_to(spec["root"])),
                    "voice": settings["voice"],
                    "voice_label": settings.get("label") or settings["voice"],
                    "gen_s": round(gen_s, 3),
                    "audio_s": round(audio_s, 3),
                    "rtf": round(gen_s / audio_s, 3) if audio_s else None,
                    "sample_rate": sample_rate,
                    "device": device,
                }
            )
    return {"engine": "xtts", "items": items}


WORKERS = {
    "piper": _worker_piper,
    "kokoro": _worker_kokoro,
    "xtts": _worker_xtts,
}


def worker_main(engine: str) -> None:
    spec = json.load(sys.stdin)
    result = WORKERS[engine](spec)
    json.dump(result, sys.stdout)


def list_xtts_speakers() -> None:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tts = _load_xtts(device)
    print(json.dumps({"device": device, "speakers": tts.speakers}, ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------


def run_engine(engine: str, python: Path) -> dict:
    if engine == "kokoro":
        ensure_kokoro_models()
    outdir = OUT_ROOT / engine
    outdir.mkdir(parents=True, exist_ok=True)
    spec = {
        "engine": engine,
        "root": str(OUT_ROOT),
        "outdir": str(outdir),
        "roles": engine_spec(engine),
        "lines": {str(n): text for n, text in LINES.items()},
    }
    started = time.perf_counter()
    process = subprocess.run(
        [str(python), str(Path(__file__).resolve()), "--worker", engine],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"{engine} worker failed ({process.returncode}): "
            f"{(process.stderr or process.stdout).strip()[-1500:]}"
        )
    result = json.loads(process.stdout.strip().splitlines()[-1])
    result["run_s"] = round(elapsed, 2)
    result["roles"] = spec["roles"]
    return result


def engine_python(engine: str) -> Path:
    if engine == "piper":
        candidate = REPO / ".venv" / "bin" / "python"
        if not candidate.is_file():
            raise SystemExit("Piper baseline needs the app virtualenv at .venv")
        return candidate
    candidate = _venv_python(engine)
    if not candidate.is_file():
        raise SystemExit(f"{engine} venv not found: {candidate}")
    return candidate


def build_index(results: list[dict], failures: dict[str, str]) -> Path:
    engines = [e for e in ("piper", "kokoro", "xtts") if any(r["engine"] == e for r in results)]
    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for result in results:
        for item in result["items"]:
            by_key.setdefault((item["role"], item["n"]), {})[result["engine"]] = item

    def header(engine: str) -> str:
        label = ENGINE_LABELS[engine]
        if engine == "xtts":
            device = next(
                (item.get("device") for result in results if result["engine"] == "xtts"
                 for item in result["items"]),
                None,
            )
            if device:
                label += f" · {device.upper()}"
        return label

    notes = [
        "Piper baseline is the voice currently configured in <code>config.json</code>.",
        "Kokoro-82M Spanish: only <code>ef_dora</code> (female) and "
        "<code>em_alex</code>/<code>em_santa</code> (male) ship with v1.0; the "
        "presidenta uses an <em>ef_dora + af_heart</em> blend (80/20 style vectors) "
        "that must be auditioned by ear.",
        "XTTS-v2 speakers are chosen by median F0 and speaking rate from the model's 58 "
        "built-in studio speakers (Claribel Dervla, Royston Min, Narelle Moon, "
        "Gilberto Mathias); confirm age/accent by ear.",
        "XTTS-v2 model license: CPML (non-commercial); this is personal rehearsal use.",
        "Generation time is the synthesis call only, model already loaded; "
        "RTF = seconds of compute per second of audio (lower is better).",
    ]

    rows = []
    for role in ROLES:
        for number, text in LINES.items():
            cells = []
            for engine in engines:
                item = by_key.get((role, number), {}).get(engine)
                if item:
                    cells.append(
                        "<td>"
                        f'<audio controls preload="none" src="{item["path"]}"></audio>'
                        f'<div class="voice">{item["voice_label"]}</div>'
                        f'<div class="time">{item["gen_s"]:.2f} s &middot; '
                        f'RTF {item["rtf"]:.2f}</div>'
                        "</td>"
                    )
                else:
                    error = failures.get(engine, "not generated")
                    cells.append(f'<td class="missing">—<div class="voice">{error}</div></td>')
            rows.append(
                f"<tr><th><span class=\"{'role-' + role}\">{ROLE_LABELS[role]}</span>"
                f"<div class=\"line\">Línea {number}: “{text}”</div></th>{''.join(cells)}</tr>"
            )

    headers = "".join(f"<th>{header(e)}</th>" for e in engines)
    note_items = "".join(f"<li>{note}</li>" for note in notes)
    failure_items = "".join(
        f"<li><strong>{ENGINE_LABELS.get(engine, engine)}</strong>: {error}</li>"
        for engine, error in failures.items()
    )
    failure_block = (
        f"<section class=\"warn\"><h2>Engines skipped</h2><ul>{failure_items}</ul></section>"
        if failure_items
        else ""
    )

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comparación de voces — simulador de jurado</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 2rem; color: #1c1c1e; background: #fafaf8; }}
  h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
  p.lead {{ color: #555; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{ border: 1px solid #dcdcd6; padding: .6rem .7rem; vertical-align: top; text-align: left; }}
  thead th {{ background: #f0efe9; font-size: .9rem; }}
  tbody th {{ font-weight: 600; min-width: 22rem; }}
  .line {{ font-weight: 400; color: #555; font-size: .82rem; margin-top: .35rem; }}
  audio {{ width: 15rem; max-width: 100%; }}
  .voice {{ font-size: .8rem; color: #444; margin-top: .35rem; }}
  .time {{ font-size: .78rem; color: #666; font-variant-numeric: tabular-nums; }}
  .missing {{ color: #a33; font-size: .85rem; }}
  .role-juez_metodologo {{ color: #7a3d8f; }}
  .role-juez_tecnico {{ color: #1d5f9e; }}
  .role-juez_presidente {{ color: #9e3d2e; }}
  .role-tutor {{ color: #2f6b46; }}
  .notes, .warn {{ background: #fff; border: 1px solid #dcdcd6; padding: 1rem 1.25rem;
                   margin-top: 1.5rem; border-radius: .4rem; }}
  .warn {{ border-color: #e0b4a8; background: #fff7f4; }}
  h2 {{ font-size: 1rem; margin: 0 0 .5rem; }}
  ul {{ margin: .25rem 0; padding-left: 1.25rem; font-size: .85rem; }}
</style>
</head>
<body>
<h1>Comparación de voces — simulador de jurado</h1>
<p class="lead">Misma línea, mismo rol, distinto motor. Escucha y elige por oído; la app
no cambia de voz hasta que decidas.</p>
<table>
<thead><tr><th>Rol y línea</th>{headers}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
{failure_block}
<section class="notes"><h2>Notas</h2><ul>{note_items}</ul></section>
</body>
</html>
"""
    index = OUT_ROOT / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="append", choices=sorted(WORKERS), default=None)
    parser.add_argument("--worker", choices=sorted(WORKERS), help=argparse.SUPPRESS)
    parser.add_argument("--list-voices", action="store_true", help="list XTTS speakers and exit")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
        return
    if args.list_voices:
        list_xtts_speakers()
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failures: dict[str, str] = {}
    for engine in args.engine or ["piper", "kokoro", "xtts"]:
        print(f"== {engine}: synthesizing…", flush=True)
        try:
            result = run_engine(engine, engine_python(engine))
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures[engine] = str(exc)
            print(f"!! {engine} failed: {exc}", file=sys.stderr, flush=True)
            continue
        results.append(result)
        avg = sum(item["gen_s"] for item in result["items"]) / len(result["items"])
        print(f"   {engine}: {len(result['items'])} wavs, avg {avg:.2f} s/line")

    timings = OUT_ROOT / "timings.json"
    timings.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    index = build_index(results, failures)
    print(f"index: {index}")
    if failures:
        print(f"skipped: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
