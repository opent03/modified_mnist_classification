'''
@author: viet
This file contains basic image processing methods
and a method to compose methods LOL
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from models import view_image
import pandas as pd
import cv2 as cv
from skimage.restoration import denoise_tv_chambolle
from PIL import Image


def to3chan(img_array):
    new_array = []
    for i in range(len(img_array)):
        e = img_array[i][0]
        new_image = [e,e,e] # 3 channels
        new_array.append(new_image)
    return np.asarray(new_array)

def threshold_background(image_array, threshold=240):
    print("Thresholding background...")
    new_array = []
    for image in image_array:
        image = (image >= threshold)*255
        new_array.append(image)
    return np.array(new_array, dtype=np.uint8)

def compose(image_array, functions):
    'Composes a bunch of bs together to make a bigger bs'
    new_array = image_array
    for f in functions:
        new_array = f(new_array)
    return np.array(new_array, dtype=np.uint8)

def thin(image_array): 
    print("Thinning image...")
    image_array = image_array.astype(np.uint8)
    new_array = []
    for image in image_array:
        new_array.append(cv.ximgproc.thinning(image, thinningType=1))
    return np.array(new_array, dtype=np.uint8)

def flatten(image_array: np.ndarray):
    print("Flattening...")
    new_array = []
    for image in image_array:
        new_array.append(image.flatten())
    return np.array(new_array)