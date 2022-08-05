import cv2
import time
import numpy as np
import HandTrackingModule as htm
from cvzone.HandTrackingModule import HandDetector

#########################
wCam,hCam = 640, 480

#########################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4,hCam)
pTime = 0

detector = HandDetector(detectionCon=0.4, maxHands=2)


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        lmList = hands[0]['lmList']
        print(lmList[4], lmList[8])

        x1,y1,z1 = lmList[4]
        x2,y2,z2 = lmList[8]
        cv2.circle(img, (x1,y1),8,(125,100,100),cv2.FILLED)
        cv2.circle(img, (x2, y2),6, (125, 100, 100), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,100,100),2)


    #define fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)

    cv2.imshow("Img", img)
    cv2.waitKey(1)