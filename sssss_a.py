import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

cap=cv2.VideoCapture(1)

while True:
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    
    if key==ord('t'):
        break