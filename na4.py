import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


def find_vessels(image, mask, window_size=5):
    # Preprocessing: smoothing and enhancing contrast
    knn = joblib.load("knn.pkl")
    features = []
    positions = []
    result = np.array(mask, copy=True, dtype=bool)
    image = image[:, :, 1]
    h, w = image.shape
    half_window = window_size // 2

    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            # Extract window from image
            window = image[
                y - half_window : y + half_window + 1,
                x - half_window : x + half_window + 1,
            ]
            value = np.mean(image[y, x])
            variance = np.var(window)
            minimum = np.min(window)
            maximum = np.max(window)

            positions.append((y, x))
            features.append([value, variance, minimum, maximum])
    predictions = knn.predict(np.array(features))
    predictions = (predictions >= 0.5).astype(bool)
    for i, (y, x) in enumerate(positions):
        result[y, x] &= predictions[i]
    return result


def extract_features(image, manual, window_size):
    features = []
    labels = []
    image = image[:, :, 1]
    h, w = image.shape
    half_window = window_size // 2

    for y in range(half_window, h - half_window, 2):
        for x in range(half_window, w - half_window, 2):
            # Extract window from image
            window = image[
                y - half_window : y + half_window + 1,
                x - half_window : x + half_window + 1,
            ]

            # Extract features (example: variance of pixel values)
            value = np.mean(image[y, x])
            variance = np.var(window)
            minimum = np.min(window)
            maximum = np.max(window)
            # Get label from mask
            label = manual[y, x]

            features.append([value, variance, minimum, maximum])
            labels.append(label)

    return np.array(features), np.array(labels)


def train_knn():

    window_size = 5
    features, labels = [], []
    for i in range(21, 31):
        # Load the image and manual mask
        image = np.array(Image.open(f"DRIVE/training/images/{i}_training.tif"))
        manual = np.array(Image.open(f"DRIVE/training/1st_manual/{i}_manual1.gif"))
        f, l = extract_features(image, manual, window_size)
        features.extend(f)
        labels.extend(l)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )

    # Initialize and train classifier
    if os.path.exists("knn.pkl"):
        knn = joblib.load("knn.pkl")
        print("Loaded model from disk")
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, "knn.pkl")

    # Predict on test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    train_knn()
