import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

#webcam

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

#hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

#find a coeff to transform our output into cm
#this will require a real distance measurement instrument
x = [310,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

coeff = np.polyfit(x,y,2)

#Loop
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        lmList = hands[0]['lmList']
        x,y,w,h = hands[0]['bbox']
        #print(lmList)
        # we can add a multiplier to adapt this measurement to any size of hand
        # it should be developed like triangulation or use another distance as a reference
        x1, y1, z1 = lmList[5]
        x2, y2, z2 = lmList[17]
        #these values represent 2 points that we need to track. mediapipe provides more details

        distance = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))
        #print(abs(x2-x1), distance)

        #use the real distance coeff
        A,B,C = coeff
        distance_cm = int(A*distance**2 + B*distance + C)
        print(f"pixel_wise distance{distance} = {distance_cm} cm")

        #add the measurement on the image
        #cvzone.putTextRect(img, f'{distance_cm}cm',(x,y))
        cv2.putText(img,f'{distance_cm}cm',(x,y),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.9, (0,5,5))



    cv2.imshow("Image", img)
    cv2.waitKey(1)


