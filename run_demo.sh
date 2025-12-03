#!/usr/bin/env bash
# Lightweight shortcut to run the interview demo with ByteTrack enabled (default)
# Usage: ./run_demo.sh [VIDEO_PATH] [OUTPUT_DIR]
# Example: ./run_demo.sh car.mp4 ./results

set -euo pipefail
VIDEO=${1:-car.mp4}
OUTDIR=${2:-./results}

# Prefer the virtualenv's python if it's present so the script works even when
# users run it with `sh run_demo.sh` (which doesn't source bash venv activators).
VENV_PY="venv/bin/python3"
if [ -x "$VENV_PY" ]; then
  PY="$VENV_PY"
else
  PY="python3"
fi

echo "Running demo with input='$VIDEO' output_dir='$OUTDIR' using python: $PY"

# Ensure output dir exists
mkdir -p "$OUTDIR"

# Run the main application
"$PY" main.py -i "$VIDEO" -o "$OUTDIR" -s "$OUTDIR/res_vid.mp4" -v

# End
