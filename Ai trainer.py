import cv2
import time
import PoseEstimationModule as pm
import numpy as np


count = 0
direction = 0
pTime = 0

cam = cv2.VideoCapture(0)
detector = pm.poseEstimation()

while True:
    success, img = cam.read()
    img = cv2.resize(img, (1000,1000))
    img = detector.findPose(img, draw=False)
    lmList = detector.positionFinder(img, draw=False)

    #fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'Fps {str(int(fps))}', (70, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

    #print(lmList)
    if len(lmList) !=0:

        #Left hand
        angle = detector.angleFinder(img, 11, 13, 15, draw=True)
        percentage = np.interp(angle, (210, 310), (0,100))
        barNum = np.interp(angle, (220, 310), (400, 150))

        #If you want to detect right hand uncomment the code below and over write it on left hand
        #Right hand
        #angle = detector.angleFinder(img, 12, 14, 16, draw=True)

  
        # check for curl
        color = (255, 0, 255)
        if percentage == 100:
            color = (0, 255, 0)
            if direction == 0:
                count += 0.5
                direction = 1
        if percentage == 0:
            color = (0, 255, 0)
            if direction ==1:
                count += 0.5
                direction = 0
        # bar
        cv2.rectangle(img, (550, 150), (620, 400), color , 3)
        cv2.rectangle(img, (550, int(barNum)), (620, 400), color, cv2.FILLED)
        cv2.putText(img, f'{str(int(percentage))} %', (550, 100), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
        # counter
        cv2.rectangle(img, (20, 255), (300, 460), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45,460),  cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)


    cv2.imshow("img", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
