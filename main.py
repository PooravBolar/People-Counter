from ultralytics import YOLO
import cvzone
import cv2
import math
from sort import *

cap = cv2.VideoCapture('video.mp4')

#cap.set(3,1280) Only for webcam
#cap.set(4,720)

model = YOLO('Yolo_weights\yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('mask.png')

# Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limitsUp = [90,188,370,332]
limitsDown = [500,254,780,399]

totalCountUp = []
totalCountDown = []

# Access video
while True:
    success,img = cap.read()

    imgRegion = cv2.bitwise_and(img,mask)
    imgGraphics = cv2.imread('graphics-1.png',cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(730,50))
    results = model(imgRegion,stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w , h = x2 -x1 , y2 - y1
            bbox = int(x1),int(y1),int(w),int(h)

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass == 'person' and conf>0.4):
                #cvzone.cornerRect(img,(x1,y1,w,h),l=10)
                #cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1-20)),offset=3,scale=0.6,thickness=1)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections) 
    
    cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(255,0,255),5)
    cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(255,0,255),5)

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w , h = x2 -x1 , y2 - y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=10,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1-20)),offset=3,scale=0.6,thickness=1)

        cx ,cy = x1+w//2 , y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limitsUp[0]<cx<limitsUp[2] and limitsUp[1]-10<cy<limitsUp[3]+10:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id) 
                cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,255,0),5)

        if limitsDown[0]<cx<limitsDown[2] and limitsDown[1]-10<cy<limitsDown[3]+10:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id) 
                cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,255,0),5)

    cv2.putText(img,str(len(totalCountUp)),(930,135),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img,str(len(totalCountDown)),(1180,135),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)

    cv2.imshow("Img",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 