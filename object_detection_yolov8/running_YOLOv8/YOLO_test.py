import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model('../img/download.jpg', show=True)
cv2.waitKey(0)