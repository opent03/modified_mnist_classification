"""
@author: viet
This program creates a test dataset to draw bounding boxes, and then
attempts to feed into our trained CNN for prediction
"""

import pickle
import pandas as pd
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import random as rng
import cv2 as cv

def load_data(dr, train_data, train_labels, sub): 
    'Loads the data into variables and returns them'
    'Copied from the other file because whenever importing it runs the other file as well hmmm'
    trf, tef = open(dr + train_data, 'rb'), open(dr + sub, 'rb')
    train_data, test_data = pickle.load(trf), pickle.load(tef)
    train_labels = pd.read_csv(dr + train_labels, sep=',')

    return train_data, train_labels, test_data

print(__doc__)

data_dir = 'data/'
train_data, train_labels, sub = 'train_images.pkl', 'train_labels.csv', 'test_images.pkl'

# Load the data into variables and normalize data
X, y, sub = load_data(data_dir, train_data, train_labels, sub)


src = X[2]
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


cv.waitKey()
cv.destroyAllWindows()


    