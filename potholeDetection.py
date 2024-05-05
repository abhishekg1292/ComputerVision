# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:22:05 2024

@author: Lenovo
"""

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='pothole.yaml', epochs=10)

model = YOLO(r'D:\ComputerVision\runs\detect\train2\weights\best.pt')
model.val()

model.predict(
   source='D:\ComputerVision\potholes.mp4',
   conf=0.25,
   save=True
)
