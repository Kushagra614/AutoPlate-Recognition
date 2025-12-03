# AutoPlate-Recognition 🚗🔎

A compact Automatic License Plate Recognition (ALPR) pipeline built around YOLOv8 (vehicle/plate detection), EasyOCR (text recognition) and an optional ByteTrack-based tracker. The project produces per-frame CSV results and can optionally write an annotated MP4 for demos.

Why this repo
- Lightweight, easy-to-run demo pipeline for detection → tracking → OCR.
- Uses a pure-Python centroid tracker as a reliable fallback when native `lap` isn't available.

Features
- ✅ Vehicle and license-plate detection (YOLOv8)
- ✍️ OCR of cropped plates using EasyOCR
- 🧭 Tracking with ByteTrack (when `lap` is installed) or centroid fallback
- 📁 Outputs per-frame CSV and optional annotated video

Quick start (local)
1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Run the quick demo (preferred):

```bash
./run_demo.sh car.mp4 ./results
```

Or run the main pipeline directly:

```bash
python main.py -i car.mp4 -o ./results --save-video ./results/annotated.mp4
```

Notes on ByteTrack / `lap` ⚠️
- ByteTrack improves multi-object tracking but depends on the native `lap` package. If `lap` can't be installed (compilation or CI constraints) the code will fall back to the centroid tracker. To use ByteTrack, install `lap` inside the activated venv; you may need system packages like `build-essential` and `python3-dev` on Debian/Ubuntu.

What to expect (outputs)
- `./results/results.csv` — per-frame records including frame number, vehicle id, vehicle bbox, plate bbox, OCR text and confidence
- `./results/annotated*.mp4` — optional annotated video(s) when `--save-video` is used

Minimal CLI reference
- `-i/--input` : path to input video file (required for `main.py`)
- `-o/--output`: output directory (defaults to `./results`)
- `--save-video`: path to an annotated MP4 to write
- `--no-bytetrack`: disable ByteTrack and force centroid tracking
- `-v/--verbose`: enable verbose logging

Pipeline (typical)
1. Detection & OCR: `python main.py -i car.mp4 -o ./results --save-video ./results/annotated.mp4`
2. (Optional) Interpolate missing detections: `python missing_data.py --input ./results/results.csv --output ./results/interpolated.csv`
3. (Optional) Produce a visualization video: `python visualize.py --csv ./results/interpolated.csv --video car.mp4 --output final_demo.mp4`

CSV columns (common)
- `frame_nmr` — frame index
- `car_id` — tracking id assigned to the vehicle
- `car_bbox` — vehicle bbox [x1, y1, x2, y2]
- `license_plate_bbox` — plate bbox [x1, y1, x2, y2]
- `license_plate_bbox_score` — detection confidence
- `license_number` — OCR text
- `license_number_score` — OCR confidence

Where to look in the code
- `main.py` — main pipeline (detection → tracking → OCR) and CLI
- `ocr_utils.py` — OCR helpers, formatting and CSV writer
- `missing_data.py` — interpolation utilities for missing frames
- `visualize.py` — visualizer that draws boxes and labels from CSV

Troubleshooting
- "model not found": ensure the detection model(s) are in `models/` as described by any model README in that folder
- `lap` fails to build: install system build tools and python dev headers, or skip ByteTrack and use the centroid fallback
- If OpenCV can't import, ensure you're using the venv python where `opencv-python` was installed

Contributing
- Fork, create a feature branch, add tests for new behavior, and open a PR

