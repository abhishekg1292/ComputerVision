# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:51:25 2024

@author: Lenovo
"""

import cv2
import time
import numpy as np

video = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputMom1.mp4', video, 30, (640, 480))

# Video capture using the webcam
cam = cv2.VideoCapture(0)

# Let the system sellp for some time before the camera starts
time.sleep(5)
background_image = 0

# Lets capture the background 
for i in range(60):
    rtrn, bgnd = cam.read()

# Flip the background image
background_image = np.flip(bgnd, axis=1)

# Now we will read frames from the webcam, the loop will break if the camera is closed
while (cam.isOpened()):
    ret, img = cam.read()
    if not ret:
        break
    
    # Flip the image
    img = np.flip(img, axis=1)
    
    #Convert the color space from RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #Generate the mask to detect the color of cloak. I am using a red colored cloak.
    lower = np.array([100, 40, 40])        
    upper = np.array([100, 255, 255]) 
    #lower = np.array([0, 30, 30])
    #upper = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)

    lower = np.array([155, 40, 40]) 
    upper = np.array([180, 255, 255]) 
    #lower = np.array([160, 30, 30])
    #upper = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower, upper)
    
    mask1 = mask1 + mask2
    
    # Open the mask image. This includes erosion followed by dilation. This removes noise.
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), dtype=np.uint8))
    
    # Dilate the mask image to improve the covered area
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), dtype=np.uint8))
    
    # Segment out the red part of the image by doing the bitwise and of the image and inverted mask1
    result1 = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask1))
    
    # Generate the background image just for the red masked region
    result2 = cv2.bitwise_and(background_image, background_image, mask=mask1)
    
    # Generate the final image by adding the two results
    finalResult = cv2.addWeighted(result1, 1, result2, 1, 0)
    out.write(finalResult)

cam.release()
out.release()