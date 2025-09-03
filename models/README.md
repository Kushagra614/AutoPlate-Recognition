# Models Directory

This directory contains the trained models for the AutoPlate Recognition system.

## Required Models

### 1. License Plate Detector Model
- **File**: `license_plate_detector.pt`
- **Description**: Custom trained YOLOv8 model for license plate detection
- **Status**: ⚠️ **MISSING** - You need to train or download this model

## How to Get the License Plate Model

### Option 1: Train Your Own Model
1. Collect and annotate license plate images
2. Use YOLOv8 training pipeline:
   ```bash
   yolo train data=path/to/dataset.yaml model=yolov8n.pt epochs=100
   ```
3. Save the trained model as `license_plate_detector.pt`

### Option 2: Use Pre-trained Model
1. Download a pre-trained license plate detection model
2. Ensure it's compatible with YOLOv8 format
3. Place it in this directory as `license_plate_detector.pt`

### Option 3: Create Placeholder (for testing)
For initial testing without a trained model, you can create a placeholder:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Use base YOLOv8 model
model.save('./models/license_plate_detector.pt')
```

## Model Requirements
- Format: PyTorch (.pt)
- Framework: YOLOv8 (Ultralytics)
- Input: RGB images
- Output: Bounding boxes for license plates

## Notes
- The vehicle detection uses the standard YOLOv8n model (`yolov8n.pt`) which is automatically downloaded
- Make sure the license plate model is trained on similar data to your use case
- Model performance depends heavily on training data quality and diversity
