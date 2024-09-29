import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(image, P=8, R=1):
    gray = np.mean(image, axis=2)  # Convert to grayscale
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    return hist / (hist.sum() + 1e-7)  # Normalize the histogram

def extract_features(images):
    return np.array([extract_lbp_features(img) for img in images])