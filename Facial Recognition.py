# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:32:26 2024

@author: Lenovo
"""

from imutils import paths
import face_recognition
import cv2
import os
import numpy as np

img_paths = list(paths.list_images(r'D:\ComputerVision\face_recognition_venv\Photos'))
encodingList = []
nameList = []

for index, img_path in enumerate(img_paths):
        
    name = img_path.split(os.path.sep)[-2]
    img = cv2.imread(img_path)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Locate faces
    boxes = face_recognition.face_locations(imgRGB)
    
    #Face Embeddings
    encodings = face_recognition.face_encodings(imgRGB, boxes)
    
    for encoding in encodings:
        encodingList.append(encoding)
        nameList.append(name)

faceData = {"faceEncodings": encodingList,
            "names": nameList}

video_capture = cv2.VideoCapture(0)

fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

video = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', video, fps, (width, height))

processInd = 1
reductionFactor = 0.25

while (video_capture.isOpened()):
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    if processInd:
    
        resizeframe = cv2.resize(frame, (0, 0), fx=reductionFactor, fy=reductionFactor)
        frameRGB = cv2.cvtColor(resizeframe, cv2.COLOR_BGR2RGB)
        
        #Locate faces
        boxes = face_recognition.face_locations(frameRGB)
        
        if len(boxes) == 0:
            out.write(frame)
            processInd = not processInd
            continue
        
        encodings = face_recognition.face_encodings(frameRGB, boxes)
        
        names = []
        
        for encoding in encodings:
            matches = face_recognition.compare_faces(faceData["faceEncodings"], encoding)
            name = "Trespasser"
            
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                for i in matchedIdxs:
                    name = faceData["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
                names.append(name)
            else:
                names.append(name)
                
    processInd = not processInd
            
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top *= (1/reductionFactor)
        right *= (1/reductionFactor)
        bottom *= (1/reductionFactor)
        left *= (1/reductionFactor)
        
        if name == "Trespasser":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
            
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (int(left), int(bottom) - 35), (int(right), int(bottom)), color, cv2.FILLED)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (int(left), int(bottom)), font, 1.0, (0, 0, 0), 1)
    out.write(frame)

video_capture.release()
out.release()
    
    
    