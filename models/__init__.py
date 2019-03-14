import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(dr, train_data, train_labels, sub): 
    'Loads the data into variables and returns them'
    'Copied from the other file because whenever importing it runs the other file as well hmmm'
    trf, tef = open(dr + train_data, 'rb'), open(dr + sub, 'rb')
    train_data, test_data = pickle.load(trf), pickle.load(tef)
    train_labels = pd.read_csv(dr + train_labels, sep=',')

    return train_data, train_labels, test_data

def view_image(image):
    'Displays a single image'
    im = np.array(image, dtype='float')
    plt.imshow(im)
    plt.show()

