import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_image(img_path, img_size=(256, 256)):
    img = Image.open(img_path).resize(img_size)
    return np.array(img) / 255.0

def load_dataset(data_dir, img_size=(256, 256)):
    images = []
    labels = []
    
    for label in ['authentic', 'forged']:
        path = os.path.join(data_dir, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img_array = load_image(img_path, img_size)
            images.append(img_array)
            labels.append(0 if label == 'authentic' else 1)
    
    return np.array(images), np.array(labels)
def split_dataset(images, labels, test_size=0.2, val_size=0.2):
    if len(images) < 3:
        print("Warning: Dataset too small to split. Using all data for training.")
        return images, images, images, labels, labels, labels

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42
    )

    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test