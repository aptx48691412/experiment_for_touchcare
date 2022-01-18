import cv2
import sys
sys.path.append('c:\python39\lib\site-packages')
from playsound import playsound
import mediapipe as mp


#cap=cv2.VideoCapture(0)


def mediapipe_detection(frame):

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    results=hands.process(frame_rgb)
    ii=results.multi_hand_landmarks
    if ii:
        for i in ii:
            mpDraw.draw_landmarks(frame,i,mpHands.HAND_CONNECTIONS)

    gray_new=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    global threshold
    ret_,threshold=cv2.threshold(gray_new,0,255,cv2.THRESH_OTSU)


cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

while True:
    ret,frame=cap.read()
    mediapipe_detection(frame)
    cv2.imshow('frame',threshold)
    key=cv2.waitKey(1)
    if key==27:
                
        playsound(r"C:\Users\imdam\0321.wav")
        break
