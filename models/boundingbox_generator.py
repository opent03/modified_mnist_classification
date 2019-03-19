"""
@author: viet
Literally does what the file name says it does
Please don't use cuz we don't talk about this in our report
"""

import pickle
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random as rng
import cv2 as cv
from models import load_data, view_image


print(__doc__)


data_dir = 'data/'
train_data, train_labels, sub = 'train_images.pkl', 'train_labels.csv', 'test_images.pkl'

# Load the data into variables and normalize data
X, y, sub = load_data(data_dir, train_data, train_labels, sub)

nb = 5
src = X[nb]
oldsrc = X[nb]
src = src/255
src = np.uint8(src)
src_gray = cv.blur(src, (3, 3))
source_window = 'Source'

cv.namedWindow(source_window, flags=cv.WINDOW_NORMAL)
#cv.imshow(source_window, src)
max_thresh = 255
threshold = 2
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
# Draw contours
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), 
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    # cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
# Show in a window
# cv.imshow('Contours', drawing)

# Find bounding box with greatest area
areas = []
for boxes in boundRect:
    areas.append(boxes[2]*boxes[3])

# Crop image
x, y, w, h = boundRect[np.argmax(areas)]
crop_image = oldsrc[y:y+h, x:x+w]
view_image(crop_image)
print(crop_image)

# Black padding
side = 28
topbot = (side-h)/2
leftright = (side-w)/2
f = np.floor
c = np.ceil
crop_image = cv.copyMakeBorder(crop_image, int(f(topbot)), 
    int(c(topbot)), int(f(leftright)), int(c(leftright)), cv.BORDER_CONSTANT, value=[0,0,0])
view_image(crop_image)
print(crop_image.reshape(-1).shape)
np.savetxt('cropped_image_4.csv', crop_image.reshape(-1), delimiter=',')
#cv.waitKey()
#cv.destroyAllWindows()


    
