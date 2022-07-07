import cv2
from cvzone.HandTrackingModule import HandDetector
import math

#webcam

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

#hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

#Loop
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        lmList= hands[0]['lmList']
        x1, y1 = lmList[5]
        x2, y2 = lmList[17] #these values represent 2 points that we need to track. mediapipe provides more details

        distance = math.sqrt((y2-y1) ** 2 + (x2-x1) ** 2)
        print(abs(x2-x1), distance)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

