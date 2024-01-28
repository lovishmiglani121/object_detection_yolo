import os
import cv2
import math
from ultralytics import YOLO

# Replace with the path to the directory containing images
image_directory = '../img/'
# output_directory = '../output/'

# Load YOLO model
model = YOLO('../running_YOLOv8/yolov8s.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "pen"]

for image_file in os.listdir(image_directory):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_directory, image_file)

        # Read the image
        img = cv2.imread(img_path)

        # Run YOLO on the image
        results = model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

                # Get confidence and class information
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'

                # Draw label
                size_label = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                create = x1 + size_label[0], y1 - size_label[1] - 3
                cv2.rectangle(img, (x1, y1), create, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Save or display the result
        output_path = os.path.join(output_directory, f'result_{image_file}')
        cv2.imwrite(output_path, img)
        cv2.imshow("Image", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
