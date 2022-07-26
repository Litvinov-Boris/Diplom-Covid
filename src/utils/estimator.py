import numpy as np

import src.segmentation.image.preprocessing as Preprocessor


def content(minor: np.ndarray, major: np.ndarray) -> float:
    """
    Calculates a content of minor array in major array in percent

    :param minor: Minor array
    :param major: Major array
    :return: A content of `minor` array in `major` array in percent
    """
    # Ensure that arrays are binarized
    m_1 = Preprocessor.binarize(minor)
    m_2 = Preprocessor.binarize(major)
    intersection = np.where((m_1 * m_2) == 1, 1, 0)

    return np.count_nonzero(intersection) / np.count_nonzero(m_2)
