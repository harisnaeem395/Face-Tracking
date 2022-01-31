# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:41:47 2022

@author: Haris
"""

import cv2
import time
import numpy as np
import time
import imutils

face_cascade = cv2.CascadeClassifier('C:/Users/Haris/OneDrive/Desktop/Project/Ball_Tracker/Ball_Tracker/haarcascade_frontalface_default.xml')
frame_rate = 30
prev = 0
gray_prev = None
p0 = []


def prep(img):
    # Your code begins here
    img= cv2.resize (img, (600,600))
    img= cv2.flip (img,1)
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Your code ends here
    return gray, img    


def get_trackable_points(gray,img):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) != 0:
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            
    
        p0 = cv2.goodFeaturesToTrack(roi_gray,maxCorners=70,qualityLevel=0.001,minDistance=5)
        p0 = p0[:, 0, :] if p0 is not None else []
        # Your code begins here
        x,y,w,h= faces[0,:]
                    
        x += 10
        y += 10
        w -= 2*10
        h -= 2*10
                    
        roi_gray = gray[y:y+h, x:x+w]
        
        if len(p0) != 0:
            p0[:,0] += x
            p0[:,1] += y
        # Your code ends here
   
    return p0, faces, img


def do_track_face(gray_prev, gray, p0):
    p1, isFound, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, 
                                                            None,
                                                            winSize=(31,31),
                                                            maxLevel=10,
                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
                                                            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                                                            minEigThreshold=0.00025)
    #your code begins here   
    #if isFound==1:
     #   valid_points=p1
    
    new_points = p1[isFound[:,0] == 1, :]
    #your code ends here
    return p1



cam = cv2.VideoCapture("C:/Users/Haris/OneDrive/Documents/Face.mp4")
p0=[]
#global p0  
if not cam.isOpened():
    raise Exception("Could not open camera/file")


#global p0
while True:
    time_elapsed = time.time() - prev
    if time_elapsed > 1./frame_rate:
        
        ret_val,img = cam.read()
        
        if not ret_val:
                cam.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
                gray_prev = None  # previous frame
                p0 = []  # previous points
                continue
        prev = time.time()
        
        gray, img = prep(img)
        
        #global p0
        
        if len(p0) <= 10: 
            p0, faces, img = get_trackable_points(gray,img)
            gray_prev = gray.copy()
        
        else:
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                p1 = do_track_face(gray_prev, gray, p0)
            for i in p1:
                cv2.drawMarker(img, (i[0], i[1]),[255,0,0],0)
            p0 = p1
                   
        cv2.imshow('Video feed', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
              
cv2.destroyAllWindows()  
