import numpy as np
import cv2
from scipy.stats import entropy
from skimage.feature import hog

# Helper untuk grayscale
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Ekstraksi fitur warna
def extract_color_features(img):
    channels = cv2.split(img) # B, G R
    features = []

    for channel in channels:
        mean_val = np.mean(channel)
        std_val = np.std(channel)

        # Untuk entropy
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.flatten()

        hist_sum = hist.sum()
        if hist_sum != 0:
            hist = hist / hist_sum
        ent_val = entropy(hist, base=2)

        # Tidak memakai skew karena dikatakan overkill

        features.extend([mean_val, std_val, ent_val])

    return features


# Ekstraksi fitur teksture
def extract_texture_features(img):
    gray = to_grayscale(img)

    features = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    return features


# Ekstraksi fitur shape
def extract_shape_features(img):
    gray = to_grayscale(img)

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        # Circularity (nice bonus feature)
        if perimeter != 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0

        return [area, perimeter, circularity]

    return [0, 0, 0]


def extract_features(img):
    color_features = extract_color_features(img)
    texture_features = extract_texture_features(img)
    shape_features = extract_shape_features(img)

    return np.hstack([
        color_features,
        texture_features,
        shape_features
    ])

