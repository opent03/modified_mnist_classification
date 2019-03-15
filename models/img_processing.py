'''
@author: viet
This file contains basic image processing methods
and a method to compose methods LOL
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv


def to3chan(img_array):
    new_array = []
    for i in range(len(img_array)):
        e = img_array[i][0]
        new_image = [e,e,e] # 3 channels
        new_array.append(new_image)
    return np.asarray(new_array)

def threshold_background(image_array, threshold=230):
    new_array = []
    for image in image_array:
        image = (image >= threshold)*255
        new_array.append(image)
    return np.asarray(new_array)

def denoising(image_array):
    new_array = []
    for image in image_array:
        trans = cv.fastNlMeansDenoising(image_array)
        new_array.append(trans)
    return np.asarray(new_array)

def compose(image_array, functions):
    'Composes a bunch of bs together to make a bigger bs'
    new_array = image_array
    for f in functions:
        new_array = f(new_array)
    return new_array

