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
import imgaug as ia
from imgaug import augmenters as iaa

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

def augment_tf_out_of_them(image_array: np.ndarray):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.GaussianBlur(sigma=(0,3.0))),
        sometimes(iaa.Crop(px=(0,6))),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        sometimes(iaa.Emboss(alpha=1, strength=0.5))
    ])
    image_array = np.transpose(image_array, (0,2,3,1))
    augmented = seq.augment_images(image_array)
    augmented = np.transpose(augmented, (0,3,1,2))
    return augmented


