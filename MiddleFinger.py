import cv2
import numpy as np
import HandTrackingModule as htm
import time
#import pyautogui
import autopy

###################################
wCam, hCam = 640,480
frameR = 100 # frame reduction ratio
smoothening = 7

###################################
pTime = 0
pLocX, pLocY = 0,0
cLocX, cLocY = 0,0


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
wScr, hScr = autopy.screen.size()

detector = htm.handDetector(maxHands=1)
while True:
    # 1: Find the fingers and landmarks

    success, img=cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:] # index finger
        x2, y2 = lmList[12][1:] #middle finger
        #print(x1,y1)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        #reframe the pointer movement surface to the screen
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only index finger: Moving mode
        if fingers[1]==1 and fingers[2]==0: # moving mode when the index finger is up and middle down
            #5 converting coordinates
            '''
            x3 = np.interp(x1,(0,wCam),(0,wScr))
            y3 = np.interp(y1,(0,hCam),(0,hScr))
            '''
            x3 = np.interp(x1,(frameR,wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR), (0,hScr))

            #6 Smoothen values
            cLocX = pLocX +(x3 -pLocX)/smoothening
            cLocY = pLocY+(y3 -pLocY)/smoothening
            #7 move the mouse
            autopy.mouse.move(wScr - x3,y3)
            cv2.circle(img,(x1,y1),15,(10,255,0),cv2.FILLED) #circle on the mouse finger
            pLocX,pLocY = cLocX,cLocY


            # 5. Convert and adapt the coordinate to any screen resolution

            # 6. Smoothen the mouse movement

            #7. Setup the clicking mode

            #8. find distance between clicking fingers
            # this can simply be done by selecting a peace sign as a click trigger,
            # but the bellow function use the distance between the index and middle finger
        if fingers[1]==1 and fingers[2]==1:
            #9 Checking the distance between 2 fingers
            length, img, lineInfo= detector.findDistance(8,12,img)
            print(length)
            if length<40:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(255,10,0),cv2.FILLED)
                autopy.mouse.click()

            #9. setup the threshold for the click action

    #10.  frame rate config
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0), 3)
    #displaying the fps on the image


    #11. display
    cv2.imshow("Image", img)
    cv2.waitKey(1)