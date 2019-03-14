'''
@author: viet
Prototyping the thinning pipeline, might not use it tho
'''
from models import load_data, view_image
import numpy as np

import cv2 as cv
from PIL import Image
train_data, train_labels, sub_data = load_data('data/', 
'train_images.pkl', 'train_labels.csv', 'test_images.pkl')


train_labels = train_labels['Category'].values          # Get labels

train_data, sub_data = (train_data)[:,:,:,None], (sub_data)[:,:,:,None]
train_data, sub_data = np.transpose(train_data, (0,3,1,2)), np.transpose(sub_data, (0,3,1,2))

def convert_to_3_channels(img_array):
    'Literally does what the name says it does'
    new_array = []
    for i in range(len(img_array)):
        e = img_array[i][0]
        new_image = [e,e,e] # 3 channels
        new_array.append(new_image)
    return np.asarray(new_array)

train_data = convert_to_3_channels(train_data)
sub_data = convert_to_3_channels(sub_data)
print(train_data.shape)
image = np.array(sub_data[0][0], dtype=np.uint8)
image = np.rint(image/255)
image = np.array(image*255, dtype=np.uint8)

#des = np.zeros(shape=(image.shape), dtype=np.uint8)
image = cv.ximgproc.thinning(image)
view_image(image)