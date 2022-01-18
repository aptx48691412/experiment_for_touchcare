import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def detect_mask(frame):
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_filter,high_filter=np.array([20,20,20]),np.array([100,100,100])
    mask=cv2.inRange(frame,low_filter,high_filter)
    bitwise_frame=cv2.bitwise_and(gray_frame, gray_frame,mask=mask)
    bitwise_frame_=cv2.bitwise_not(frame,frame, mask=mask)
    bitwise_frame__=cv2.bitwise_and(frame,frame,mask)
    return mask,bitwise_frame,bitwise_frame_,bitwise_frame__


cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    bitwise_frame=detect_mask(frame)
    cv2.imshow('frame',bitwise_frame[0])
    cv2.imshow('frame_bitwise_and',bitwise_frame[1])
    cv2.imshow('frame_bitwise_and_',bitwise_frame[2])
    #cv2.imshow('frame_bitwise_and_',bitwise_frame[3])
    key=cv2.waitKey(1)
    if key==27:
        break
      