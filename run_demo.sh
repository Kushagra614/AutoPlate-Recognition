#!/usr/bin/env bash
# Lightweight shortcut to run the interview demo with ByteTrack enabled (default)
# Usage: ./run_demo.sh [VIDEO_PATH] [OUTPUT_DIR]
# Example: ./run_demo.sh car.mp4 ./results

set -euo pipefail
VIDEO=${1:-car.mp4}
OUTDIR=${2:-./results}

# If a venv is present in the project root, activate it
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

python3 main.py -i "$VIDEO" -o "$OUTDIR" -s "$OUTDIR/res_vid.mp4" -v

# End
