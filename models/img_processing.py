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
    ia.seed(np.random.randint(0))
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-15, 15),
            shear=(-5, 5)
        )
        ], random_order=True) # apply augmenters in random order
    image_array = np.transpose(image_array, (0,2,3,1))
    augmented = seq.augment_images(image_array)
    augmented = np.transpose(augmented, (0,3,1,2))
    return augmented


