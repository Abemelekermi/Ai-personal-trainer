import cv2
import mediapipe as mp
import time
import math
class poseEstimation():
    def __init__(self, 
                 mode=False,
               complexity=1,
               smooth =True,
               enable_segmentation=False,
               smooth_segmentation=True,
               detection_confidence=0.5,
               tracking_confidence=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence


        self.mpdraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, 
                                     self.enable_segmentation, self.smooth_segmentation, self.detection_confidence, 
                                     self.tracking_confidence)
    def findPose(self, img, draw=True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.Results = self.pose.process(imgrgb)
        if self.Results.pose_landmarks:
            if draw:
                self.mpdraw.draw_landmarks(img, self.Results.pose_landmarks, 
                                        self.mpPose.POSE_CONNECTIONS)
        return img

    def positionFinder(self, img, draw=True):
        self.lmlist = []
        if self.Results.pose_landmarks:
            for id,lm in enumerate(self.Results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmlist
    def angleFinder(self, img, p1, p2, p3, draw=True):
        #landMarks
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]
    
        #angles
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        
        if angle < 0:
            angle += 360
        #print(angle)

        if draw:
            #circles
            cv2.circle(img, (x1,y1), 9, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 15, (255, 0, 255), 2)
            cv2.circle(img, (x2,y2), 9, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (255, 0, 255), 2)
            cv2.circle(img, (x3,y3), 9, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3,y3), 15, (255, 0, 255), 2)

            #Lines
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(img, (x2, y2), (x2, y2), (255, 0, 0), 2)
            #cv2.line(img, (x3, y3), (x2, y2), (255, 0, 0), 2)

            #TExt
            cv2.putText(img, str(int(angle)), (x2-20, y2), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)

        return angle


def main():
    cam = cv2.VideoCapture("Resources\Vid2.mp4")
    pTime = 0
    detector = poseEstimation()
    while True:
        success, img=cam.read()
        img = detector.findPose(img)
        lmlist = detector.positionFinder(img, draw=False)
        if len(lmlist) != 0:
            print(lmlist[12])
            cv2.circle(img, (lmlist[14][1],lmlist[14][2]), 15, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)
        cv2.imshow("img",img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()