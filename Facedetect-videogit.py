# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:51:27 2020

@author: crist
"""

import cv2
import matplotlib.pyplot as plt


datasetPath='/Users/crist/Documents/PYTHON/GITHUB_proj\dataset'

##########Declare CASCADE CLASSIFIERS#######################
###Load require xml classifiers for face,eye and smile #####
facePath = "/Users/crist/Documents/PYTHON/GITHUB_proj/HARACascade Git/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
eyePath = "/Users/crist/Documents/PYTHON/GITHUB_proj/HARACascade Git/opencv/data/haarcascades/haarcascade_eye.xml"
smilePath = "/Users/crist/Documents/PYTHON/GITHUB_proj/HARACascade Git/opencv/data/haarcascades/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)


# Read the image
img = cv2.imread('revdec1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(20,10))
plt.imshow(gray, cmap='gray')
plt.show()

# Detect faces
## scaleFactor parameter specifying how much the image size is reduced at each image scale
##minNeighbors parameter specifying how many neighbors each candidate rectangle should have to retain it
faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,
                                     )
# For each face draw rectangle around the face
for (x, y, w, h) in faces: 
    cv2.rectangle(gray, (x, y), (w+x, h+y), color=(0,0,0),thickness=20) 
    
plt.figure(figsize=(20,10))
plt.imshow(gray, cmap='gray')
plt.show()


#######################Video Capturing#############################
# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 (default value)
video_cap=cv2.VideoCapture(0) 
#Save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outputvideo.avi', fourcc,15.0,(640,480))
#Normal size sans-serif font
font = cv2.FONT_HERSHEY_SIMPLEX 


while True:
    # Capture the frame
    succ, imgf = video_cap.read()
    gray = cv2.cvtColor(imgf, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

    # Drawing a rectangle around the faces
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(imgf, (fx, fy), (fx+fw, fy+fh), (255,70,120), thickness=3)
        roi_gray = gray[fy:fy+fh,fx:fx+fw] 
        roi_color = imgf[fy:fy+fh, fx:fx+fw] 
        cv2.putText(imgf,'MyFace',(fx, fy), font,fontScale=1,color=(255,70,120),thickness=2)
   
        
    eyes = eyeCascade.detectMultiScale(roi_gray,minNeighbors=15)
    
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0, 255, 0),2)
        cv2.putText(imgf,'Eyes',(fx + ex,fy + ey), 1, 1, (0, 255, 0), 1)
                    

    smiles = smileCascade.detectMultiScale(roi_gray,scaleFactor= 1.14,minNeighbors=40)
                                           
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 80, 0), 2)
        cv2.putText(imgf,'Smile',(fx + sx,fy + sy), 1, 1, (255, 80, 0),1)

    
    cv2.putText(imgf,'Number of Faces detected: ' + str(len(faces)),(30, 30), font, 1,(0,0,0),2)      
    
    # Display the frame
    cv2.imshow('Video', imgf)
    out.write(imgf)
#exit key
    if cv2.waitKey(1) & 0xFF == ord('e'):
      break

# Release the capture when everything is done 
video_cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
    
