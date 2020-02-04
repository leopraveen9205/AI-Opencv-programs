# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:54:09 2019

@author: Raja1
"""
import numpy as np
import cv2,time

video = cv2.VideoCapture(0)

check, frame = video.read()

print(check)
print(frame)

time.sleep(3)

cv2.imshow("capture", frame)

cv2.waitKey(0)

video.release()

cv2.destroyAllWindows()

################################### Capturing the video
import cv2,time

video = cv2.VideoCapture(0)

a = 1

while True:
    a = a + 1
    check, frame = video.read()
    print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("capture", gray)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
   
print(a)
video.release()
cv2.destroyAllWindows()





















