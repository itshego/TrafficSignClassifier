from ultralytics import YOLO
import cv2
import time
import os

def crop_object(frame, box, original_shape):
    orig_height, orig_width = original_shape[:2]
    current_height, current_width = frame.shape[:2]

    # Calculate scaling factors
    width_scale = orig_width / current_width
    height_scale = orig_height / current_height

    # Scale the bounding box coordinates
    x1, y1, x2, y2 = [int(coord * width_scale) if i % 2 == 0 else int(coord * height_scale) for i, coord in
                      enumerate(box.xyxy[0])]

    # Ensure coordinates are within the original frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(orig_width, x2), min(orig_height, y2)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def save_cropped_object(cropped_object, class_name, frame_number, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = f"{frame_number}_{class_name}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, cropped_object)
    print(f"Saved: {filepath}")

model = YOLO('best.pt')
cap = cv2.VideoCapture("C:\\Users\\Hego\\Desktop\\zed2_rgb_video.mp4")
output_folder = "cropped_objects"

# Get original video resolution
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_shape = (orig_height, orig_width)

frame_number = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame_number += 1
        results = model(frame, conf=0.4)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = model.names[cls]
                text = ""
                try:
                    cropped_object, original_coords = crop_object(frame, box, original_shape)

                    text = class_name  # tsc.detect_and_process_sign(cropped_object, class_name)
                    print("Yolo:", class_name)
                    save_cropped_object(cropped_object, class_name, frame_number, output_folder)

                except Exception as e:
                    print(e)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

    # Optional: little latency to slow down the video
    time.sleep(0.1)
cap.release()
cv2.destroyAllWindows()