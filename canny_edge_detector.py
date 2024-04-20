# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 00:42:15 2024

@author: Abhishek Gupta
"""

import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# Function to convert the colored image into a grayscale image
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    return gray

# Define a gauss kernel to blur the image and reduce the noise
def gauss_kernel(size = 2, sigma = 1):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normalizing_factor = 1./2.*np.pi*sigma**2
    gk = normalizing_factor*np.exp(-(x**2+y**2)/(2.*sigma**2))
    return gk

# This function calculates the gradient magnitude and direction at every pixel
def gradient_calculation(img):
    # Define Horizontal Sorbel Kernels
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
    
    # Define Vertical Sorbel Kernels
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], np.float32)
    
    # Calculate horizontal gradient
    Ix = convolve(img, Kx)
    
    # Calculate vertical gradient
    Iy = convolve(img, Ky)
    
    # Calculate the gradient magnitude and rescale from 0 to 255
    G_mag = Ix**2 + Iy**2
    G_mag = (G_mag/G_mag.max())*255
    
    # Calculate the gradient directions
    G_theta = np.arctan2(Iy, Ix)
    
    return G_mag, G_theta

# Funtion to suppress the non maximum pixels
def maximum_suppression(mag, theta):
    m, n = mag.shape
    img = np.zeros((m,n), dtype=np.int32)
    angle = theta*180.0/np.pi
    angle[angle<0] = angle[angle<0] + 180
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            a = angle[i, j]
            p1 = 255
            p2 = 255
            
            # angle = 0 deg
            if (0.0 <= a <= 22.5) or (157.5 <= a <= 180.0):
                p1 = mag[i, j+1]
                p2 = mag[i, j-1]
            
            # angle = 45 deg
            elif (22.5 <= a <= 67.5):
                p1 = mag[i-1, j+1]
                p2 = mag[i+1, j-1]
            
            # angle = 90 deg
            elif (67.5 < a <= 112.5):
                p1 = mag[i-1, j]
                p2 = mag[i+1, j]
            
            # angle = 135 deg
            elif (112.5 < a < 157.5):
                p1 = mag[i-1, j-1]
                p2 = mag[i+1, j+1]
            
            if (mag[i, j] >= p1) and (mag[i, j] >= p2):
                img[i, j] = mag[i, j]
            else:
                img[i, j] = 0
    return img

# Function for double thresholding
def thresholding(img, high, low, strong_pixel, weak_pixel):
    finalImage = np.zeros(img.shape, dtype=np.int32)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] >= high:
                finalImage[i, j] = strong_pixel
            elif img[i, j] <= low:
                finalImage[i, j] = 0
            else:
                finalImage[i, j] = weak_pixel
    return finalImage

def hysteresis(img, strong_pixel, weak_pixel):
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] == weak_pixel:
                if ((img[i+1, j-1] == strong_pixel) or
                    (img[i+1, j] == strong_pixel) or 
                    (img[i+1, j+1] == strong_pixel) or
                    (img[i, j-1] == strong_pixel) or
                    (img[i, j+1] == strong_pixel) or
                    (img[i-1, j-1] == strong_pixel) or
                    (img[i-1, j] == strong_pixel) or 
                    (img[i-1, j+1] == strong_pixel)):
                    img[i, j] = strong_pixel
                else:
                    img[i, j] = 0
    return img

# Read the image
img = plt.imread(r'C:\Users\Lenovo\Desktop\pikachu.png')

plt.imshow(img)
plt.show()

# Convert the image into grayscale
img = rgb2gray(img)
plt.imshow(img, cmap='gray')
plt.show()

# Apply the gauss kernel and reduce the noise
gk = gauss_kernel()
img = convolve(img, gk)
plt.imshow(img, cmap='gray')
plt.show()

# Calculate the gradient magnitude and direction
G_mag, G_theta = gradient_calculation(img)
plt.imshow(G_mag, cmap='gray')
plt.show()

# Perform non maximum suppression to reduce the thickness of the images
img = maximum_suppression(G_mag, G_theta)
plt.imshow(img, cmap='gray')
plt.show()

# Perform double thresholding to determine the strong and weak pixels
high = 0.1*img.max()
low = 0.005*img.max()
strong_pixel = 255
weak_pixel = 25
img = thresholding(img, high, low, strong_pixel, weak_pixel)
plt.imshow(img, cmap='gray')
plt.show()

# Additional operation to get the connectd edge
img = hysteresis(img, strong_pixel, weak_pixel)
plt.imshow(img, cmap='gray')
plt.show()