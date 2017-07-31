import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.examples.tutorials.mnist import input_data

MNIST_DIR = "data/MNIST"
USPS_DIR = "data/USPS/usps_train.jf";
mnist = input_data.read_data_sets(MNIST_DIR, one_hot = True)

def load_MNIST_data_labeled():
    return mnist.test.images, mnist.test.labels

def load_MNIST_data_unlabeled():
    return mnist.train.images, mnist.train.labels
         
def load_USPS_data():
    # read USPS data and reshape them into 28x28 size.
    USPS_file = open(USPS_DIR)
    USPS_lines = USPS_file.readlines()

    data = np.zeros((len(USPS_lines) - 2, 784))
    label = np.zeros((len(USPS_lines) - 2, 10))
    
    index = 0
    for line in USPS_lines:
        if len(line) <= 10:
            continue
        expend = line.split(' ')
        label[index, int(expend[0])] = 1
        for tmp in range(16):
            data[index, (28 * (6 + tmp) + 6):(28 * (6 + tmp) + 6 + 16)] = expend[1 + 16 * tmp:1 + 16 * (tmp + 1)]
        index = index + 1

    data = data / 2.0
    return data, label

# if __name__ == "__main__":
#     load_MNIST_data_labeled()
#     load_MNIST_data_unlabeled()
#     load_USPS_data()
