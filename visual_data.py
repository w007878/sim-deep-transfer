import numpy as np
import cv2

def show_data(batch_size, data, path=None):
    # assume data is a numpy array with size [batch_size, 784]
    # in the image, each row has 10 digits.
    tmp = data.reshape(batch_size / 10, 10, 28, 28)
    tmp = np.transpose(tmp, (0, 2, 1, 3))
    tmp = tmp.reshape(batch_size / 10 * 28, 280)
    img = np.zeros((batch_size /10 * 28, 280, 3))
    img[:, :, 0] = tmp
    img[:, :, 1] = tmp
    img[:, :, 2] = tmp

    # whether show or save image
    if not path:
        cv2.imshow('image', img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(path, img * 256)
