from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to grayscale
image = Image.open('./image27.png').convert('L')  # 'L' mode for grayscale

# Convert image to numpy array
image_np = np.array(image)

# Define number of quantization levels
num_levels = 4  # Change this to the desired number of levels

# Normalize the image (values between 0 and 1)
image_normalized = image_np / 255.0

# Quantize the image by defining bins and scaling
quantized_image = np.floor(image_normalized * num_levels) * (255 / num_levels)

# Convert the array back to uint8
quantized_image = quantized_image.astype(np.uint8)

# Display the original and quantized image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_np, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f'Quantized Image ({num_levels} levels)')
plt.imshow(quantized_image, cmap='gray')

plt.show()
