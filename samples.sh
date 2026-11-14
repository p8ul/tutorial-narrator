#!/usr/bin/env bash
# Generate the same sentence with every male English voice into ./samples/<voice>.wav
# Usage: ./samples.sh [--verbose] ["Your own sentence."]
# Quiet by default (speak.py hides library warnings); --verbose shows them.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1

EXTRA=()
if [[ "${1:-}" == "--verbose" ]]; then
  EXTRA=(--verbose)
  shift
fi

TEXT="${1:-The quick brown fox jumps over the lazy dog. This is how I sound reading a longer sentence aloud.}"
VOICES=(am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck
        bm_daniel bm_fable bm_george bm_lewis)

mkdir -p samples
for v in "${VOICES[@]}"; do
  .venv/bin/python speak.py "$TEXT" --voice "$v" --out "samples/$v.wav" ${EXTRA[@]+"${EXTRA[@]}"}
done
echo "Done. Play one with: afplay samples/am_michael.wav"
