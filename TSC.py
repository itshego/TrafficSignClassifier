import cv2
import numpy as np
import os
from tqdm import tqdm
from config import *
from stats_manager import StatsManager

def create_mask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower), np.array(upper))

def create_navy_blue_mask(img):
    return create_mask(img, LOWER_BLUE, UPPER_BLUE)

def create_red_mask(img):
    mask1 = create_mask(img, LOWER_RED1, UPPER_RED1)
    mask2 = create_mask(img, LOWER_RED2, UPPER_RED2)
    return mask1 + mask2

def adjust_ellipse(ellipse, scale_factor=1, shift_factor=0.00):
    (x, y), (width, height), angle = ellipse
    new_width = width * scale_factor
    new_height = height * scale_factor
    shift_x = width * shift_factor
    shift_y = height * shift_factor
    return (x + shift_x, y + shift_y), (new_width, new_height), angle

def validate_ellipse_size(ellipse, img_shape, threshold=0.3):
    img_height, img_width = img_shape[:2]
    (_, _), (ellipse_width, ellipse_height), _ = ellipse
    width_ratio = ellipse_width / img_width
    height_ratio = ellipse_height / img_height
    return not (abs(1 - width_ratio) > threshold or abs(1 - height_ratio) > threshold)

def mask_outside_ellipse(img, ellipse):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)
    masked_img = img.copy()
    masked_img[mask == 0] = [0, 0, 255]
    return masked_img, mask

def count_pixels(img, mask, color_type='blue'):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color_type == 'blue':
        color_mask = cv2.inRange(hsv, np.array(LOWER_BLUE), np.array(UPPER_BLUE))
    else:
        mask1 = cv2.inRange(hsv, np.array(LOWER_RED1), np.array(UPPER_RED1))
        mask2 = cv2.inRange(hsv, np.array(LOWER_RED2), np.array(UPPER_RED2))
        color_mask = mask1 + mask2
    return np.sum(cv2.bitwise_and(color_mask, mask) > 0)

def _determine_direction(pixel_counts, sign_name):
    left_top_percentage = pixel_counts[LEFT_TOP]["percentage"]
    right_top_percentage = pixel_counts[RIGHT_TOP]["percentage"]
    left_bot_percentage = pixel_counts[LEFT_BOTTOM]["percentage"]
    right_bot_percentage = pixel_counts[RIGHT_BOTTOM]["percentage"]
    left_percentage = (left_top_percentage + left_bot_percentage) / 2
    right_percentage = (right_top_percentage + right_bot_percentage) / 2

    if sign_name in ("ileri_sol_mecburi", "ileri_sag_mecburi"):
        return LEFT if left_top_percentage > right_top_percentage else RIGHT
    elif sign_name in ("ileriden_saga", "ileriden_sola", "sag_mecburi", "sol_mecburi"):
        return LEFT if left_percentage > right_percentage else RIGHT
    elif sign_name in ("sagdan_gidin", "soldan_gidin"):
        return RIGHT if left_bot_percentage > left_top_percentage else LEFT
    elif sign_name in ("saga_donulmez", "sola_donulmez"):
        return RIGHT if ((left_top_percentage + right_bot_percentage) / 2) > ((left_bot_percentage + right_top_percentage) / 2) else LEFT
    return ERRORS['invalid_sign_name']

def visualize_quadrants(img, quadrants, pixel_counts, debug_dir):
    h, w = img.shape[:2]
    result_img = np.ones((500, 400, 3), dtype=np.uint8) * 240
    scale_factor = 500 / max(h, w)

    for name, points in quadrants:
        scaled_points = (points * scale_factor).astype(np.int32)
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.fillPoly(result_img, [scaled_points], color, cv2.LINE_AA)

        text_position = scaled_points.mean(axis=0).astype(int)
        text_position[0] = max(text_position[0] - 60, 10)

        total = pixel_counts[name]["total"]
        blue = pixel_counts[name]["blue"]
        percentage = pixel_counts[name]["percentage"]

        cv2.putText(result_img, f"T:{total} B:{blue}",
                    (text_position[0], text_position[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(result_img, f"%{percentage:.1f}",
                    (text_position[0], text_position[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return result_img

def detect_ellipse(mask, img, debug_dir):
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, ERRORS['no_sign_detected']

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:
        return None, None, ERRORS['invalid_sign']

    ellipse = cv2.fitEllipse(largest_contour)
    ellipse_img = img.copy()
    cv2.ellipse(ellipse_img, ellipse, (0, 255, 0), 2)

    if not validate_ellipse_size(ellipse, img.shape):
        return None, None, ERRORS['invalid_ellipse']

    return ellipse, ellipse_img, None

def detect_and_process_sign(img, debug_dir, traffic_sign):
    if img is None:
        return ERRORS['image_read_error']

    save_debug_image(img, "1_original", debug_dir)
    navy_mask = create_navy_blue_mask(img)
    save_debug_image(navy_mask, "2_navy_blue_mask", debug_dir)

    ellipse, ellipse_img, error = detect_ellipse(navy_mask, img, debug_dir)
    if error:
        return error
    save_debug_image(ellipse_img, "3_detected_ellipse", debug_dir)

    masked_img, ellipse_mask = mask_outside_ellipse(img, ellipse)
    save_debug_image(masked_img, "4_masked_image", debug_dir)

    h, w = masked_img.shape[:2]
    center, (width, height), angle = ellipse

    quadrants = [
        (LEFT_TOP, np.array([(0, 0), (center[0], 0), center, (0, center[1])], dtype=np.int32)),
        (RIGHT_TOP, np.array([(center[0], 0), (w, 0), (w, center[1]), center], dtype=np.int32)),
        (LEFT_BOTTOM, np.array([(0, center[1]), center, (center[0], h), (0, h)], dtype=np.int32)),
        (RIGHT_BOTTOM, np.array([center, (w, center[1]), (w, h), (center[0], h)], dtype=np.int32))
    ]

    pixel_counts = {}
    for name, points in quadrants:
        quadrant_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(quadrant_mask, [points], 255)
        quadrant_mask = cv2.bitwise_and(quadrant_mask, ellipse_mask)

        total_pixels = np.sum(quadrant_mask == 255)
        blue_pixels = count_pixels(masked_img, quadrant_mask, 'blue')
        blue_percentage = (blue_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        pixel_counts[name] = {
            "total": total_pixels,
            "blue": blue_pixels,
            "percentage": blue_percentage
        }

    result_img = visualize_quadrants(masked_img, quadrants, pixel_counts, debug_dir)
    save_debug_image(result_img, "5_result", debug_dir)

    for name, counts in pixel_counts.items():
        print(OUTPUT['region_stats']['region'].format(name))
        print(OUTPUT['region_stats']['total_pixels'].format(counts['total']))
        print(OUTPUT['region_stats']['blue_pixels'].format(counts['blue']))
        print(OUTPUT['region_stats']['blue_percentage'].format(counts['percentage']))

    return _determine_direction(pixel_counts, traffic_sign)

def read_image(image_path):
    try:
        img_array = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img if img is not None else None
    except Exception as e:
        print(OUTPUT['image_read_error'].format(image_path, str(e)))
        return None

def save_debug_image(img, name, debug_dir):
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), img)

def process_folder(folder_path, debug_base_dir):
    stats_manager = StatsManager()
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    for file_name in tqdm(image_files, desc=OUTPUT['progress']['processing_files']):
        file_path = os.path.join(folder_path, file_name)
        img = read_image(file_path)
        stats_manager.increment_total_processed()

        if img is not None:
            debug_dir = os.path.join(debug_base_dir, os.path.splitext(file_name)[0])
            result = detect_and_process_sign(img, debug_dir, "ileri_sag_mecburi")
            stats_manager.add_result(file_name, result)
        else:
            stats_manager.add_result(file_name, ERRORS['image_read_error'])

    stats_manager.calculate_total_score()
    return stats_manager

def main():
    folder_path = DEFAULT_FOLDER_PATH
    output_path = os.path.join(folder_path, OUTPUT_FOLDER)
    os.makedirs(output_path, exist_ok=True)

    debug_base_dir = os.path.join(folder_path, DEBUG_FOLDER)
    os.makedirs(debug_base_dir, exist_ok=True)

    output_file_path = os.path.join(output_path, OUTPUT_FILE_NAME)
    stats_manager = process_folder(folder_path, debug_base_dir)
    print(OUTPUT['debug_images_saved'].format(debug_base_dir))
    stats_manager.write_to_file(output_file_path)

if __name__ == "__main__":
    main()
