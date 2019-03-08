"""
@author: viet
This program draws bounding boxes around numbers in the modified-MNIST
"""

import pickle
import pandas as pd
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import cv2 as cv

def load_data(dr, train_data, train_labels, sub):
    'Loads the data into variables and returns them'
    trf, tef = open(dr + train_data, 'rb'), open(dr + sub, 'rb')
    train_data, test_data = pickle.load(trf), pickle.load(tef)
    train_labels = pd.read_csv(dr + train_labels, sep=',')

    return train_data, train_labels, test_data

def view_image(image):
    'Displays a single image'
    assert image.shape == (64, 64)
    im = np.array(image, dtype='float')
    print(image[0])
    plt.imshow(im, cmap='gray')
    plt.show()

print(__doc__)

data_dir = 'data/'
train_data, train_labels, sub = 'train_images.pkl', 'train_labels.csv', 'test_images.pkl'

# Load the data into variables and normalize data
X, y, sub = load_data(data_dir, train_data, train_labels, sub)
X, sub = X/255, sub/255

# Draw bounding boxes
# We're going to work with the first image
image = X[0]
#image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = cv.blur(image, (2,2))
source_window = 'Source'
cv.namedWindow(source_window)
cv.resizeWindow(source_window, 600, 600)
cv.imshow(source_window, image)
cv.waitKey()
cv.destroyAllWindows()
