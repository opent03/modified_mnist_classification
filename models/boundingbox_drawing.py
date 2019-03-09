"""
@author: viet
This program draws bounding boxes around numbers in the modified-MNIST
"""

import pickle
import pandas as pd
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import random as rng
import cv2 as cv
from models import load_data, view_image


print(__doc__)


data_dir = 'data/'
train_data, train_labels, sub = 'train_images.pkl', 'train_labels.csv', 'test_images.pkl'

# Load the data into variables and normalize data
X, y, sub = load_data(data_dir, train_data, train_labels, sub)

rng.seed(12345)

# Test some opencv bs to see if it works
def thresh_callback(val):
    threshold = val
    canny_output = cv.Canny(src_gray, threshold, threshold*2)
    _, contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
   
    # Bounding rectangles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    print(boundRect)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), 
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        # cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    # Show in a window
    cv.imshow('Contours', drawing)

src = cv.imread('data/mnistsample.png')
#src = X[2]
#src = src/255
#src = np.uint8(src)
#print(type(src))
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#src_gray = cv.blur(src, (3, 3))
src_gray = cv.blur(src_gray, (3, 3))
#view_image(src_gray)

source_window = 'Source'

cv.namedWindow(source_window, flags=cv.WINDOW_NORMAL)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 2 #initial threshold
cv.createTrackbar('Canny Thresh: ', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()
cv.destroyAllWindows()


    