import ast
import argparse
import os

import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def visualize_results(csv_path, video_path, output_path):
    """Visualize license plate detection results on video."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found")
        return
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return
    
    results = pd.read_csv(csv_path)
    
    # load video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        car_data = results[results['car_id'] == car_id]
        if len(car_data) == 0:
            continue
            
        max_ = np.amax(car_data['license_number_score'])
        best_detection = car_data[car_data['license_number_score'] == max_]
        
        if len(best_detection) == 0:
            continue
            
        license_plate[car_id] = {
            'license_crop': None,
            'license_plate_number': best_detection['license_number'].iloc[0]
        }
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_detection['frame_nmr'].iloc[0])
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        try:
            bbox_str = best_detection['license_plate_bbox'].iloc[0]
            x1, y1, x2, y2 = ast.literal_eval(bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            
            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if license_crop.size > 0:
                license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
                license_plate[car_id]['license_crop'] = license_crop
        except Exception as e:
            print(f"Error processing license plate for car {car_id}: {e}")
            continue


    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                try:
                    # draw car
                    car_bbox_str = df_.iloc[row_indx]['car_bbox']
                    car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                                line_length_x=200, line_length_y=200)
    
                    # draw license plate
                    plate_bbox_str = df_.iloc[row_indx]['license_plate_bbox']
                    x1, y1, x2, y2 = ast.literal_eval(plate_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
    
                    # crop license plate
                    car_id = df_.iloc[row_indx]['car_id']
                    if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
                        license_crop = license_plate[car_id]['license_crop']
                        H, W, _ = license_crop.shape
        
                        try:
                            frame[int(car_y1) - H - 100:int(car_y1) - 100,
                                  int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
        
                            frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                                  int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)
        
                            (text_width, text_height), _ = cv2.getTextSize(
                                license_plate[car_id]['license_plate_number'],
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                17)
        
                            cv2.putText(frame,
                                        license_plate[car_id]['license_plate_number'],
                                        (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        4.3,
                                        (0, 0, 0),
                                        17)
        
                        except Exception as e:
                            print(f"Error overlaying license plate: {e}")
                            pass
                except Exception as e:
                    print(f"Error processing frame {frame_nmr}, row {row_indx}: {e}")
                    continue
    
            out.write(frame)
    
    out.release()
    cap.release()
    print(f"Visualization saved to: {output_path}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize license plate detection results')
    parser.add_argument('--csv', required=True, help='Path to CSV file with detection results')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', default='./output_visualization.mp4', help='Output video path')
    args = parser.parse_args()
    
    visualize_results(args.csv, args.video, args.output)


if __name__ == '__main__':
    main()
