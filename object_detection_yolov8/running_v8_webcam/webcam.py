import math

from ultralytics import YOLO
import cv2

cap=cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

model = YOLO('../running_YOLOv8/yolov8s.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "pen"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            '''as here we are getting value in the form of tensors which is not detectable if we
             want to create the bounding boxes'''

            # print(x1,y1,x2,y2)
            x1,y1,x2,y2 = int(x1),int(y1), int(x2),int(y2)
            '''now it is converted into the integers so that we can create the bounding boxes'''
            # print(x1,y1,x2,y2)
            '''we will create a rectangle here at each of the object which is detected 
                x1,y1 is the starting points and x2, y2 are the ending points '''
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255),3)
            '''lets check the confidence value now '''
            conf = math.ceil((box.conf[0]*100))/100
            '''now lets will check wich class it belongs to '''
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            size_label = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            # print(size_label)
            create = x1 + size_label[0], y1 - size_label[1] -3
            cv2.rectangle(img, (x1,y1), create, [255,0,255], -1, cv2.LINE_AA)
            cv2.putText(img, label , (x1,y1-2),0 ,1, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

    # out.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('1'):
        break
out.release()