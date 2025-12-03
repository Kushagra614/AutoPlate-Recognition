import logging
import cv2
from ultralytics import YOLO
from util import get_car, read_license_plate, write_csv
from config_loader import Config
from pathlib import Path
import argparse
import os
import sys
import numpy as np

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)


class CentroidTracker:
    """A tiny centroid-based tracker to avoid heavy dependencies like `lap`.

    Matches detections across frames by nearest centroid within a threshold.
    This is simple and works well for short interview demos.
    """
    def __init__(self, max_distance=60, max_missing=30):
        self.next_id = 1
        self.objects = {}  # id -> centroid (x,y)
        self.last_seen = {}  # id -> frames since last seen
        self.max_distance = max_distance
        self.max_missing = max_missing

    def update(self, boxes):
        """Assign IDs to the provided boxes and return list of IDs in same order.

        boxes: iterable of [x1,y1,x2,y2]
        """
        centroids = []
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids.append((cx, cy))

        if len(self.objects) == 0:
            ids = []
            for c in centroids:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.last_seen[oid] = 0
                ids.append(oid)
            return ids

        # Match existing objects to new centroids greedily
        existing_ids = list(self.objects.keys())
        existing_centroids = [self.objects[i] for i in existing_ids]

        ids = [-1] * len(centroids)
        assigned_existing = set()
        for i, c in enumerate(centroids):
            best_dist = float('inf')
            best_j = None
            for j, ec in enumerate(existing_centroids):
                if existing_ids[j] in assigned_existing:
                    continue
                d = (c[0]-ec[0])**2 + (c[1]-ec[1])**2
                if d < best_dist:
                    best_dist = d
                    best_j = j

            if best_j is not None and best_dist <= (self.max_distance ** 2):
                oid = existing_ids[best_j]
                ids[i] = oid
                assigned_existing.add(oid)
                self.objects[oid] = centroids[i]
                self.last_seen[oid] = 0
            else:
                # new object
                oid = self.next_id
                self.next_id += 1
                ids[i] = oid
                self.objects[oid] = centroids[i]
                self.last_seen[oid] = 0

        # increment last_seen for unassigned existing objects
        for oid in existing_ids:
            if oid not in assigned_existing:
                self.last_seen[oid] = self.last_seen.get(oid, 0) + 1
                if self.last_seen[oid] > self.max_missing:
                    # remove stale object
                    del self.objects[oid]
                    del self.last_seen[oid]

        return ids


class LicensePlateDetector:
    def __init__(self, config_path=None, use_bytetrack=False):
        self.vehicle_detector = None
        self.plate_detector = None
        self.logger = None
        self.results = {}
        self.config = Config(config_path) if config_path else Config()

        self.setup_logging()
        self.load_models()
        # simple tracker to avoid optional native deps like 'lap'
        self.centroid_tracker = CentroidTracker()
        # whether to use ultralytics built-in tracker (ByteTrack) if available
        self.use_bytetrack = use_bytetrack

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
                # plate model is optional for the demo: we'll fallback to heuristic crop + OCR
                self.logger.warning(f"License plate model not found at {plate_model_path}; falling back to heuristic plate cropping + OCR")
                self.plate_detector = None
            else:
                self.logger.info(f"Loading license plate detection model: {plate_model_path}")
                self.plate_detector = YOLO(plate_model_path)

            self.logger.info("Models loaded (vehicle detector ready; plate detector optional)")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def track_vehicles(self, frame):
        """Detect vehicles in the frame and return (bbox, id) pairs.

        If `use_bytetrack` is True, attempt to use ultralytics' built-in tracker
        (ByteTrack). If that fails or is not enabled, fall back to per-frame
        detection + centroid-based matching.
        """
        # Prefer ultralytics tracker when requested
        if getattr(self, 'use_bytetrack', False):
            try:
                results = self.vehicle_detector.track(frame, persist=True, tracker="bytetrack.yaml")
                if results and results[0].boxes is not None and getattr(results[0].boxes, 'id', None) is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
                    return list(zip(boxes, track_ids))
            except Exception as e:
                # If tracker fails (e.g., lap not installed), log and fall back
                self.logger.error(f"ByteTrack tracker error: {e}. Falling back to detection+centroid.")

        # Fallback: per-frame detection + centroid tracker
        try:
            results = self.vehicle_detector(frame)[0]
            boxes = []
            if hasattr(results, 'boxes') and results.boxes is not None:
                for b in results.boxes.data.tolist():
                    if len(b) < 6:
                        continue
                    x1, y1, x2, y2, score, cls = b
                    if int(cls) in VEHICLE_CLASSES and float(score) > 0.3:
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])

            if len(boxes) == 0:
                return []

            ids = self.centroid_tracker.update(boxes)
            return list(zip(boxes, ids))
        except Exception as e:
            self.logger.error(f"Error detecting vehicles: {e}")
            return []

    def process_license_plates(self, frame, vehicle_tracks, frame_number):
        """Detect and process license plates in the frame."""
        self.results[frame_number] = {}
        
        if not vehicle_tracks:
            return
        # If a dedicated plate detector model is available, use it; otherwise
        # fall back to a heuristic: crop bottom ~35% of each vehicle bbox and OCR that.
        try:
            if self.plate_detector is not None:
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

                        # Save detection
                        self.results[frame_number][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': plate_text,
                                'text_score': plate_score,
                                'bbox_score': score
                            }
                        }
            else:
                # Heuristic: for each detected vehicle, crop bottom portion and OCR
                for bbox, car_id in vehicle_tracks:
                    x1, y1, x2, y2 = map(int, bbox)
                    h = y2 - y1
                    plate_y1 = int(y2 - 0.35 * h)
                    plate_y2 = y2
                    plate_x1 = x1
                    plate_x2 = x2

                    plate_y1 = max(0, plate_y1)
                    plate_y2 = min(frame.shape[0], plate_y2)
                    plate_x1 = max(0, plate_x1)
                    plate_x2 = min(frame.shape[1], plate_x2)

                    crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                    if crop.size == 0:
                        continue

                    plate_text, plate_score = read_license_plate(crop)

                    self.results[frame_number][car_id] = {
                        'car': {'bbox': [x1, y1, x2, y2]},
                        'license_plate': {
                            'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                            'text': plate_text,
                            'text_score': plate_score,
                            'bbox_score': None
                        }
                    }
        except Exception as e:
            self.logger.error(f"Error processing license plates in frame {frame_number}: {e}")

    def process_video(self, input_path, output_dir, display=False, save_video_path=None):
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
        writer = None
        
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

                # If requested, draw bounding boxes (vehicles + plates) for display or saving
                if display or save_video_path:
                    annot = frame.copy()

                    # draw vehicle boxes and track ids
                    for item in vehicle_tracks:
                        try:
                            box, tid = item
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(annot, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(annot, f'ID:{int(tid)}', (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        except Exception:
                            continue

                    # draw plate boxes and OCR text from results if available
                    if frame_number in self.results:
                        for v in self.results[frame_number].values():
                            plate = v.get('license_plate')
                            if plate:
                                try:
                                    bx1, by1, bx2, by2 = map(int, plate.get('bbox', [0,0,0,0]))
                                    text = plate.get('text', '') or ''
                                    conf = plate.get('text_score', 0.0)
                                    cv2.rectangle(annot, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                                    cv2.putText(annot, f'{text} ({conf:.2f})', (bx1, max(0, by1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                except Exception:
                                    continue

                    # initialize video writer lazily when we have the frame shape
                    if save_video_path and writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        h, w = annot.shape[:2]
                        try:
                            writer = cv2.VideoWriter(save_video_path, fourcc, fps if fps and fps>0 else 30.0, (w, h))
                        except Exception as e:
                            self.logger.error(f'Failed to create VideoWriter: {e}')

                    # write annotated frame to output video if requested
                    if writer is not None:
                        try:
                            writer.write(annot)
                        except Exception as e:
                            self.logger.error(f'Failed to write frame to video: {e}')

                    try:
                        if display:
                            cv2.imshow('LicensePlate - Press q to quit', annot)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.logger.info('User requested exit via keypress')
                                break
                    except Exception:
                        # In headless environments cv2.imshow may fail; ignore and continue
                        pass

                frame_number += 1
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during video processing: {e}")
            return False
        finally:
            cap.release()
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
        
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
    parser.add_argument('-i', '--input', required=True, help='Path to input video file')
    parser.add_argument('-o', '--output', help='Output directory for results (default: ./results)')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    parser.add_argument('-d', '--display', action='store_true', help='Show realtime annotated video (press q to quit)')
    parser.add_argument('-s', '--save-video', help='Path to save annotated MP4 (e.g. results/annotated.mp4)')
    # Use ByteTrack by default for better tracking; provide a --no-bytetrack flag to disable it
    parser.add_argument('--no-bytetrack', action='store_false', dest='use_bytetrack', help='Disable ByteTrack and use centroid tracker instead')
    parser.set_defaults(use_bytetrack=True)
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    try:
        # Load configuration
        detector = LicensePlateDetector(args.config, use_bytetrack=args.use_bytetrack)

        # Set output directory from config if not provided (default: ./results)
        output_dir = args.output or detector.config.get('results.save_path', './results')

        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        success = detector.process_video(args.input, output_dir, display=args.display, save_video_path=args.save_video)

        if success:
            print(f"\n✅ Processing completed successfully!")
            print(f"📁 Results saved to: {output_dir}")
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