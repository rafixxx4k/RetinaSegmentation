import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def find_vessels(retina_image, mask):
    # Preprocessing: smoothing and enhancing contrast
    retina_image = retina_image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    retina_image = clahe.apply(retina_image)

    # Adaptive thresholding to segment vessels
    binary_image = cv2.adaptiveThreshold(
        retina_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Morphological operations to enhance vessel segmentation
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    result = np.logical_and(np.logical_not(binary_image), mask)
    return result
