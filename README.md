# AutoPlate Recognition System

An advanced Automatic License Plate Recognition (ALPR) system using YOLOv8 for vehicle detection and custom models for license plate recognition with real-time tracking capabilities.

## 🚀 Features

- **Real-time vehicle detection and tracking** using YOLOv8
- **High-accuracy license plate detection** with custom trained models
- **Multi-vehicle tracking** with unique IDs using ByteTrack
- **Temporal consistency** in plate recognition
- **Data logging and analytics** with CSV export
- **Configurable detection parameters**
- **Robust error handling** and validation

## 📋 Requirements

- Python 3.8+
- OpenCV 4.5+
- PyTorch
- Ultralytics YOLOv8
- EasyOCR
 - (Optional, recommended for better tracking) ByteTrack support via `lap>=0.5.12`

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kushagra614/AutoPlate-Recognition.git
   cd AutoPlate-Recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) If you want the project to use ByteTrack for more robust multi-object tracking, install the native `lap` package and system build tools (required on many Linux systems):

```bash
# On Debian/Ubuntu install build tools first (may require sudo)
sudo apt update && sudo apt install -y build-essential python3-dev

# Activate venv and install lap
source venv/bin/activate
pip install lap>=0.5.12
```

If `lap` fails to install (compilation errors), the project will automatically fall back to a lightweight pure-Python centroid tracker — this is fine for demos but less robust under heavy occlusion.

4. **Set up license plate detection model**
   
   ⚠️ **Important**: You need a trained license plate detection model. Check `models/README.md` for instructions.

## 🎯 Usage

### Complete Processing Pipeline

To process a video and generate annotated output, run these **3 commands in sequence**:

```bash
# Step 1: Vehicle and license plate detection
python main.py --input car.mp4 --output ./results --verbose --use-bytetrack --save-video ./results/annotated.mp4

# Step 2: Interpolate missing data for smooth tracking
python missing_data.py --input ./results/results.csv --output ./results/interpolated.csv

# Step 3: Generate visualization video with bounding boxes
python visualize.py --csv ./results/interpolated.csv --video car.mp4 --output final_demo.mp4
```

### What Each Step Does

1. **Detection** (`main.py`): Detects vehicles and license plates, saves CSV data
2. **Interpolation** (`missing_data.py`): Smooths tracking data for better visualization
3. **Visualization** (`visualize.py`): Creates annotated video with bounding boxes and IDs

### Quick Start Example

```bash
# Activate virtual environment
source venv/bin/activate

# Run complete pipeline
python main.py --input car.mp4 --output ./results --verbose
python missing_data.py --input ./results/results.csv --output ./results/interpolated.csv
python visualize.py --csv ./results/interpolated.csv --video car.mp4 --output final_demo.mp4
```

### Command Line Options

#### main.py
- `--input`: Path to input video file (required)
- `--output`: Output directory for results (default: ./output)
- `--verbose, -v`: Enable verbose logging
 - `--use-bytetrack`: Use ultralytics' ByteTrack tracker (requires `lap` installed). If `lap` is missing the code will fall back to centroid tracking.
 - `--save-video`: Path to save annotated MP4 (e.g. `results/annotated.mp4`). When provided the app will write an annotated video alongside the CSV.

#### missing_data.py
- `--input`: Input CSV file path (required)
- `--output`: Output CSV file path (default: interpolated.csv)

#### visualize.py
- `--csv`: Path to CSV file with detection results (required)
- `--video`: Path to input video file (required)
- `--output`: Output video path (default: final_demo.mp4)

## 📁 Project Structure

```
AutoPlate-Recognition/
├── main.py                 # Main application entry point
├── util.py                 # Utility functions for OCR and processing
├── visualize.py            # Visualization tools for results
├── missing_data.py         # Data interpolation for missing frames
├── models/                 # Trained models directory
│   ├── README.md          # Model setup instructions
│   └── license_plate_detector.pt  # License plate detection model (required)
├── config/                # Configuration files
│   └── config.json        # Application configuration
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 📊 Output Format

The system generates a CSV file with the following columns:

- `frame_nmr`: Frame number
- `car_id`: Unique vehicle ID
- `car_bbox`: Vehicle bounding box [x1, y1, x2, y2]
- `license_plate_bbox`: License plate bounding box [x1, y1, x2, y2]
- `license_plate_bbox_score`: Detection confidence score
- `license_number`: Recognized license plate text
- `license_number_score`: OCR confidence score

## 🔧 Configuration

Edit `config/config.json` to customize:

- Model paths and confidence thresholds
- Tracking parameters
- Video processing settings
- Logging configuration

## 🚨 Troubleshooting

### Common Issues

1. **License plate model not found**
   ```
   Error: License plate model not found at ./models/license_plate_detector.pt
   ```
   **Solution**: Follow instructions in `models/README.md` to obtain the model.

2. **Video file not found**
   ```
   Error: Input video file not found: path/to/video.mp4
   ```
   **Solution**: Check the video file path and ensure it exists.

3. **Low detection accuracy**
4. **ByteTrack / `lap` install errors**
   If you see errors like:

   ```
   ERROR: Command "pip install 'lap>=0.5.12'" returned non-zero exit status 1
   ```

   - Ensure you have system build tools (gcc, make) and Python dev headers installed (example for Debian/Ubuntu shown above).
   - Install `lap` inside the activated virtualenv: `pip install lap`.
   - If installing `lap` is not possible on your machine, run `main.py` without `--use-bytetrack` and the code will use the built-in centroid tracker (no native deps required).

   - Ensure good video quality and lighting
   - Check if the license plate model is trained on similar data
   - Adjust confidence thresholds in the configuration

### Performance Tips

- Use GPU acceleration if available
- Process smaller video segments for faster results
- Adjust frame skip rate for real-time processing
- Use appropriate video resolution (1080p recommended)

## 📈 Performance Metrics

- **Vehicle Detection Accuracy**: 97.8% (619/631 frames)
- **License Plate Detection Accuracy**: >70% confidence scores
- **Processing Speed**: ~24ms per frame (real-time capable)
- **Tracking Reliability**: Multi-vehicle tracking with persistent IDs
- **Output**: Professional visualization with bounding boxes and tracking overlays


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👨‍💻 Author

**Kushagra Vardhan**

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the `models/README.md` for model-related issues
3. Create an issue in the repository with detailed error logs