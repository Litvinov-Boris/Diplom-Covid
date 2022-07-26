import cv2 as cv
import numpy as np

from src.utils import Const


def enhance(image: np.ndarray):
    clahe = cv.createCLAHE(clipLimit=7.0)  # Set the contrast threshold 3.0
    img = np.uint8(image.copy() * 255)
    res = clahe.apply(img)

    return res.reshape(res.shape[0], res.shape[1], 1)


def binarize(mask):
    binarized = np.where(mask > Const.MASK_THRESHOLD, 1.0, 0.0)

    return binarized.reshape((binarized.shape[0], binarized.shape[1], 1))


def resize(image):
    return cv.resize(image, dsize=(Const.IMG_SIZE, Const.IMG_SIZE), interpolation=cv.INTER_CUBIC)


def normalize(image):
    x_max, x_min = image.max(), image.min()

    return (image - x_min) / (x_max - x_min)
