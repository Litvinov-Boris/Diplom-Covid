import os

import cv2
import numpy as np

import src.utils.const as Const
import src.segmentation.image.preprocessing as Preprocessor


def load_sample(file_path) -> np.ndarray:
    sample = Preprocessor.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) / 255.)
    return sample.reshape((Const.IMG_SIZE, Const.IMG_SIZE, 1))


def load_samples(folder_path):
    samples = []

    file_paths = os.listdir(folder_path)

    for i in range(len(file_paths)):
        path = os.path.join(folder_path, file_paths[i])
        samples.append(load_sample(path))

    return np.array(samples)
