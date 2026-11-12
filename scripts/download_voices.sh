#!/bin/sh
# Download the Piper Spanish voices used in config.json (offline neural TTS).
set -eu
DIR="${JURADO_PIPER_VOICES:-$HOME/Library/Application Support/JuradoDefensa/voices}"
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/es
mkdir -p "$DIR"
for v in es_MX/claude/high/es_MX-claude-high es_MX/ald/medium/es_MX-ald-medium \
         es_AR/daniela/high/es_AR-daniela-high es_ES/davefx/medium/es_ES-davefx-medium \
         es_ES/sharvard/medium/es_ES-sharvard-medium; do
  n=$(basename "$v")
  [ -f "$DIR/$n.onnx" ] || curl -fL -o "$DIR/$n.onnx" "$BASE/$v.onnx"
  [ -f "$DIR/$n.onnx.json" ] || curl -fL -o "$DIR/$n.onnx.json" "$BASE/$v.onnx.json"
done
echo "Voices in $DIR"
