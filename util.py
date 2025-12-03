import easyocr
import cv2
import numpy as np
import string

# Lazy-initialized OCR reader to avoid heavy work at import time
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def preprocess_plate(plate_img):
    """
    Preprocesses a license plate image to improve OCR accuracy.
    - Converts to grayscale
    - Applies contrast enhancement (CLAHE)
    - Resizes to a fixed height for consistency
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Resize to a fixed height for more consistent OCR
    target_height = 64
    aspect_ratio = enhanced_gray.shape[1] / enhanced_gray.shape[0]
    target_width = int(target_height * aspect_ratio)
    resized_plate = cv2.resize(enhanced_gray, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    return resized_plate


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    # Preprocess the image to improve OCR accuracy
    preprocessed_plate_crop = preprocess_plate(license_plate_crop)

    reader = get_reader()
    detections = reader.readtext(preprocessed_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    Uses proximity-based association instead of strict containment.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of tuples containing (bbox, track_id) where bbox is [x1, y1, x2, y2].

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    # Calculate license plate center
    plate_center_x = (x1 + x2) / 2
    plate_center_y = (y1 + y2) / 2
    
    best_car_idx = -1
    min_distance = float('inf')
    
    for j in range(len(vehicle_track_ids)):
        bbox, car_id = vehicle_track_ids[j]
        xcar1, ycar1, xcar2, ycar2 = bbox
        
        # Method 1: Check if license plate overlaps with vehicle (relaxed)
        overlap_x = max(0, min(x2, xcar2) - max(x1, xcar1))
        overlap_y = max(0, min(y2, ycar2) - max(y1, ycar1))
        
        if overlap_x > 0 and overlap_y > 0:
            best_car_idx = j
            break
            
        # Method 2: Check if license plate is near the vehicle (proximity)
        car_center_x = (xcar1 + xcar2) / 2
        car_center_y = (ycar1 + ycar2) / 2
        
        # Calculate distance between centers
        distance = ((plate_center_x - car_center_x) ** 2 + (plate_center_y - car_center_y) ** 2) ** 0.5
        
        # Check if plate is within reasonable distance of car
        car_width = xcar2 - xcar1
        car_height = ycar2 - ycar1
        max_distance = max(car_width, car_height) * 1.5  # Allow 1.5x car size distance
        
        if distance < max_distance and distance < min_distance:
            min_distance = distance
            best_car_idx = j

    if best_car_idx != -1:
        bbox, car_id = vehicle_track_ids[best_car_idx]
        xcar1, ycar1, xcar2, ycar2 = bbox
        return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1
