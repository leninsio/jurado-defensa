"""Long-lived XTTS-v2 worker, run with the XTTS engine venv.

The app starts this once and keeps it alive so the 2 GB model loads a single
time. Protocol: one JSON object per line on stdin
``{"text": str, "speaker": str, "out": str, "language": "es"}``; one JSON line
per request on stdout: ``{"ok": true, "out": path, "seconds": float}`` or
``{"ok": false, "error": str}``. The first stdout line is
``{"ready": true, "device": ..., "speakers": [...]}``.

XTTS-v2 is licensed under the Coqui Public Model License (non-commercial),
which fits this personal rehearsal app.
"""

from __future__ import annotations

import json
import os
import sys
import time
import wave
from pathlib import Path

ENGINE_DIR = Path(
    os.environ.get(
        "JURADO_XTTS_HOME",
        Path.home() / "Library" / "Application Support" / "JuradoDefensa" / "engines" / "xtts",
    )
)


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    data = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (data * 32767).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def main() -> int:
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("TTS_HOME", str(ENGINE_DIR / "tts_home"))
    os.environ.setdefault("HF_HOME", str(ENGINE_DIR / "hf_home"))
    # Keep library chatter off stdout: it carries the protocol.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    import torch
    from TTS.api import TTS

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    sample_rate = tts.synthesizer.output_sample_rate
    sys.stdout = real_stdout
    _emit({"ready": True, "device": device, "speakers": list(tts.speakers or [])})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            started = time.perf_counter()
            sys.stdout = sys.stderr
            try:
                samples = tts.tts(
                    text=request["text"],
                    speaker=request["speaker"],
                    language=request.get("language", "es"),
                )
            finally:
                sys.stdout = real_stdout
            out = Path(request["out"])
            _write_wav(out, samples, sample_rate)
            _emit({"ok": True, "out": str(out), "seconds": round(time.perf_counter() - started, 3)})
        except Exception as exc:  # noqa: BLE001 - report every failure to the app
            sys.stdout = real_stdout
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
