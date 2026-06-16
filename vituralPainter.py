# https://github.com/ysthehurricane/Virtual-Drawing/blob/main/virtual_painter.ipynb
# https://www.kaggle.com/code/ysthehurricane/4-incredible-projects-using-mediapipe-must-watch

import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class handDetector:
    def __init__(self, static_mode=False, maxhands=2, detection_confident= 0.5, tracking_confident=0.5):
        self.static_mode = static_mode
        self.maxhands = maxhands
        self.detection_confident = detection_confident
        self.tracking_confident = tracking_confident
        model_path = 'gesture_recognizer.task'
        
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2  # Maximum number of hands to detect
        )
        self.top_idx = [4,8,12,16,20]
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

        
        
    def findhands(self, frame, draw_landmark=True):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        self.detection_result = self.recognizer.recognize(mp_image)
        
        if self.detection_result.hand_landmarks:
            for hand_landmarks in self.detection_result.hand_landmarks:
                for id, lm in enumerate(hand_landmarks):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # Custom logic for specific landmarks
                    if id == 4: # Tip of the thumb
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    else:
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        
        return frame
    
    def gethandlocation(self, frame, handNo=0, draw_landmark=True):
        self.lmlist = []
        if self.detection_result.hand_landmarks:
            myhand = self.detection_result.hand_landmarks[handNo] 
            
            for idx,lm in enumerate(myhand):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                
                self.lmlist.append([idx,cx,cy])
                if draw_landmark:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        return self.lmlist
    
    def fingercheck(self):
        fingers = []
        
        if self.lmlist[self.top_idx[0]][1] > self.lmlist[self.top_idx[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for idx in range(1,5):
            if self.lmlist[self.top_idx[idx]][2] < self.lmlist[self.top_idx[idx]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
widthCam, heightCam = 1210, 720

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)

handdetect = handDetector(detection_confident=0.8)
drawingcolor = (0,0,0)
xp = 0
yp = 0
brushthickness = 10
eraserthickness = 50
canvasimg = np.zeros((720, 1280, 3), np.uint8)

while True:
    
    check, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (10,10),(100,100),(0,255,0),cv2.FILLED)
    cv2.rectangle(frame, (120,10),(210,100),(0,255,255),cv2.FILLED)
    cv2.rectangle(frame, (230,10),(320,100),(255,255,0),cv2.FILLED)
    cv2.rectangle(frame, (340,10),(430,100),(100,255,200),cv2.FILLED)
    
    cv2.rectangle(frame, (450,10),(540,100),(130,0,200),cv2.FILLED)
    cv2.rectangle(frame, (560,10),(650,100),(200,120,120),cv2.FILLED)
    cv2.rectangle(frame, (670,10),(760,100),(130,210,100),cv2.FILLED)
    cv2.rectangle(frame, (780,10),(870,100),(180,210,200),cv2.FILLED)
    
    cv2.rectangle(frame, (890,10),(980,100),(135,200,50),cv2.FILLED)
    cv2.rectangle(frame, (1000,10),(1090,100),(80,80,230),cv2.FILLED)
    cv2.rectangle(frame, (1110,10),(1240,100),(0,0,0),cv2.FILLED)
    cv2.putText(frame, "Eraser", (1120,60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    
    
    frame = handdetect.findhands(frame, draw_landmark=True)
    lmlist = handdetect.gethandlocation(frame, draw_landmark=False)
    
    if len(lmlist) != 0:
        
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        
        fingers = handdetect.fingercheck()
        
        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            
            if 10 <= y2 <= 100:
                if 10 <= x2 <= 100:
                    drawingcolor = (0,255,0)
                elif 120 <= x2 <= 210:
                    drawingcolor = (0,255,255)
                elif 230 <= x2 <= 320:
                    drawingcolor = (255,255,0)
                elif 340 <= x2 <= 430:
                    drawingcolor = (100,255,200)
                elif 450 <= x2 <= 540:
                    drawingcolor = (130,0,200)
                elif 560 <= x2 <= 650:
                    drawingcolor = (200,120,120)
                elif 670 <= x2 <= 760:
                    drawingcolor = (130,210,100)
                elif 780 <= x2 <= 870:
                    drawingcolor = (180,210,200)
                elif 890 <= x1 <= 980:
                    drawingcolor = (135,200,50)
                elif 1000 <= x2 <= 1090:
                    drawingcolor = (80,80,230)
                elif 1110 <= x2 <= 1240:
                    drawingcolor = (0,0,0)
                
                
            cv2.rectangle(frame, (x1,y1-20),(x2,y2+20),drawingcolor,cv2.FILLED)
                
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (x1,y1),15, drawingcolor,cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
                
            if brushthickness == (0,0,0):
                cv2.line(frame, (xp, yp),(x1, y1), drawingcolor, eraserthickness)
                cv2.line(canvasimg, (xp, yp),(x1, y1), drawingcolor, eraserthickness)
            else:
                cv2.line(frame, (xp, yp),(x1, y1), drawingcolor, brushthickness)
                cv2.line(canvasimg, (xp, yp),(x1, y1), drawingcolor, brushthickness)
            
            xp, yp = x1, y1
    
    grayimg = cv2.cvtColor(canvasimg, cv2.COLOR_BGR2GRAY)
    _, invimg = cv2.threshold(grayimg, 50,255, cv2.THRESH_BINARY_INV)
    invimg = cv2.cvtColor(invimg, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(frame, invimg)
    img = cv2.bitwise_or(img, canvasimg)
    
    cv2.imshow('Virtual Painter', img)
    #cv2.imshow('Virtual Painter canvas', canvasimg)
    #cv2.imshow('Virtual Painter invert', invimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()