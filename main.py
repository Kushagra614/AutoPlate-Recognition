import logging
import cv2
from ultralytics import YOLO
from util import get_car, read_license_plate, write_csv
from config_loader import Config
from pathlib import Path
import argparse
import os
import sys


class LicensePlateDetector:
    def __init__(self, config_path=None):
        self.vehicle_detector = None
        self.plate_detector = None
        self.logger = None
        self.results = {}
        self.config = Config(config_path) if config_path else Config()

        self.setup_logging()
        self.load_models()

    def setup_logging(self):
        """Configure logging based on configuration."""
        log_config = self.config.get_logging_config()
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if specified
        log_path = log_config.get('save_path')
        if log_path:
            Path(log_path).mkdir(parents=True, exist_ok=True)

    def load_models(self):
        """Load YOLO models for vehicle and license plate detection."""
        try:
            # Load vehicle detection model
            vehicle_config = self.config.get_model_config('vehicle_detection')
            vehicle_model_path = vehicle_config.get('model_path', 'yolov8n.pt')
            
            self.logger.info(f"Loading vehicle detection model: {vehicle_model_path}")
            self.vehicle_detector = YOLO(vehicle_model_path)
            
            # Load license plate detection model
            plate_config = self.config.get_model_config('license_plate')
            plate_model_path = plate_config.get('model_path', './models/license_plate_detector.pt')
            
            if not os.path.exists(plate_model_path):
                self.logger.error(f"License plate model not found at {plate_model_path}")
                self.logger.error("Please check the models/README.md for instructions on obtaining the model")
                raise FileNotFoundError(f"License plate model not found: {plate_model_path}")
            
            self.logger.info(f"Loading license plate detection model: {plate_model_path}")
            self.plate_detector = YOLO(plate_model_path)
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def track_vehicles(self, frame):
        """Tracks vehicles in the frame using YOLO's built-in tracker."""
        try:
            # Use the built-in tracker from ultralytics
            results = self.vehicle_detector.track(frame, persist=True, tracker="bytetrack.yaml")

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                # Get bounding boxes, track IDs
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Combine boxes and track_ids for easy lookup
                return list(zip(boxes, track_ids))
        except Exception as e:
            self.logger.error(f"Error tracking vehicles: {e}")
        
        return []

    def process_license_plates(self, frame, vehicle_tracks, frame_number):
        """Detect and process license plates in the frame."""
        self.results[frame_number] = {}
        
        if not vehicle_tracks:
            return
            
        try:
            plates = self.plate_detector(frame)[0]
            
            if plates.boxes is None or len(plates.boxes) == 0:
                return
                
            for plate in plates.boxes.data.tolist():
                if len(plate) < 6:
                    continue
                    
                x1, y1, x2, y2, score, _ = plate
                
                # Validate bounding box coordinates
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                    continue
                    
                # Find the car associated with this plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2, score, _), vehicle_tracks)
                
                if car_id != -1:
                    # Validate plate region
                    h, w = frame.shape[:2]
                    if int(y2) > h or int(x2) > w or int(y1) < 0 or int(x1) < 0:
                        continue
                        
                    # Extract and process the license plate
                    plate_img = frame[int(y1):int(y2), int(x1):int(x2), :]
                    
                    if plate_img.size == 0:
                        continue
                        
                    plate_text, plate_score = read_license_plate(plate_img)
                    
                    # Debug logging
                    self.logger.debug(f"Frame {frame_number}, Car {car_id}: OCR result = '{plate_text}', score = {plate_score}")
                    
                    min_confidence = self.config.get('ocr.min_confidence', 0.5)
                    
                    # Save detection even if OCR fails (for debugging)
                    if True:  # Always save for now to see what's happening
                        self.results[frame_number][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': plate_text,
                                'text_score': plate_score,
                                'bbox_score': score
                            }
                        }
        except Exception as e:
            self.logger.error(f"Error processing license plates in frame {frame_number}: {e}")

    def process_video(self, input_path, output_dir):
        """Process the input video, perform detection and tracking, and save results."""
        # Validate input file
        if not os.path.exists(input_path):
            self.logger.error(f"Input video file not found: {input_path}")
            return False
            
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"Error opening video file: {input_path}")
            return False

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.logger.info(f"Processing video: {total_frames} frames at {fps:.2f} FPS")
        
        frame_number = 0
        processed_frames = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % 30 == 0:  # Log progress every 30 frames
                    self.logger.info(f"Processing frame {frame_number}/{total_frames} ({frame_number/total_frames*100:.1f}%)")
                
                # Validate frame
                if frame is None or frame.size == 0:
                    frame_number += 1
                    continue
                    
                # Track vehicles in the current frame
                vehicle_tracks = self.track_vehicles(frame)
                
                if vehicle_tracks:
                    # Process license plates for the detected vehicles
                    self.process_license_plates(frame, vehicle_tracks, frame_number)
                    processed_frames += 1

                frame_number += 1
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during video processing: {e}")
            return False
        finally:
            cap.release()
        
        self.logger.info(f"Processed {processed_frames} frames with detections out of {frame_number} total frames")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write results to CSV
        csv_path = output_path / 'results.csv'
        try:
            write_csv(self.results, str(csv_path))
            self.logger.info(f"Results saved to {csv_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='License Plate Detection and Recognition')
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    try:
        # Load configuration
        detector = LicensePlateDetector(args.config)
        
        # Set output directory from config if not provided
        output_dir = args.output or detector.config.get('results.save_path', './output')
        
        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        success = detector.process_video(args.input, output_dir)
        
        if success:
            print(f"\n✅ Processing completed successfully!")
            print(f"📁 Results saved to: {args.output}")
            print(f"📊 Check results.csv for detection data")
        else:
            print(f"\n❌ Processing failed. Check the logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(0)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"\n❌ Application error: {e}")
        sys.exit(0)
        
if __name__ == '__main__':
    main()