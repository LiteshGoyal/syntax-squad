import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_dataset, split_dataset
from src.feature_extraction import extract_features
from src.model import ImageForgeryDetector
from src.utils import plot_images

# Load and preprocess the dataset
data_dir = 'data/raw'
images, labels = load_dataset(data_dir)

if len(images) == 0:
    print("Error: No images found in the dataset. Please check your data directory.")
    sys.exit(1)

print(f"Total images loaded: {len(images)}")

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Extract features
X_train_features = extract_features(X_train)
X_val_features = extract_features(X_val)
X_test_features = extract_features(X_test)

# Train the model
model = ImageForgeryDetector()
model.train(X_train_features, y_train)

# Evaluate the model
accuracy, report = model.evaluate(X_test_features, y_test)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Visualize some results
test_predictions = model.predict(X_test_features)
plot_images(X_test, y_test, test_predictions)