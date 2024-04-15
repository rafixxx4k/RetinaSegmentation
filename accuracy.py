import numpy as np
from collections import Counter


def accuracy(image1, image2):
    """
    Calculates the accuracy, sensitivity, and specificity of two binary images.

    Parameters:
    image1 (ndarray): The ground truth binary image.
    image2 (ndarray): The predicted binary image.

    Returns:
    tuple: A tuple containing the accuracy, sensitivity, and specificity values.
    """
    score = [[0, 0], [0, 0]]
    # [[true negative, false positive],
    # [false negative, true positive]]
    image1 = np.array(image1, dtype=bool)
    image2 = np.array(image2, dtype=bool)
    for i, j in zip(image1, image2):
        score[i][j] += 1
    accuracy = (score[0][0] + score[1][1]) / sum(sum(score, []))
    sensitivity = score[1][1] / sum(score[1])
    specificity = score[0][0] / sum(score[0])
    print(f"{accuracy:.2f}\t{sensitivity:.2f}\t{specificity:.2f}")
    return accuracy, sensitivity, specificity
