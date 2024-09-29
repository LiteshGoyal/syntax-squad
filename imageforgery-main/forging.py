import cv2
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from tensorflow import keras
# from tensorflow.keras import layers
import os

# Function to convert image to grayscale
def convert_to_grayscale(image_path, output_path=None):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Save the grayscale image if an output path is specified
    if output_path:
        cv2.imwrite(output_path, grayscale_image)
        print(f"Grayscale image saved at: {output_path}")

    return grayscale_image

# Apply DCT to an 8x8 block
def apply_dct_block(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Apply inverse DCT to an 8x8 block
def apply_idct_block(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Quantize the DCT coefficients of an 8x8 block
def quantize_block(block, quant_matrix):
    return np.round(block / quant_matrix)

# Dequantize the DCT coefficients of an 8x8 block
def dequantize_block(block, quant_matrix):
    return block * quant_matrix

# Zigzag function to extract significant coefficients from an 8x8 block
def zigzag_scan_block(block):
    rows, cols = block.shape
    vector = []
    for sum_idx in range(rows + cols - 1):
        if sum_idx % 2 == 0:
            for i in range(min(sum_idx, rows - 1), max(-1, sum_idx - cols), -1):
                vector.append(block[i][sum_idx - i])
        else:
            for i in range(max(0, sum_idx - cols + 1), min(rows, sum_idx + 1)):
                vector.append(block[i][sum_idx - i])
    return vector

# Function to process an image in 8x8 blocks
def process_image_in_blocks(image, quant_matrix, block_size=8):
    h, w = image.shape
    dct_img = np.zeros_like(image, dtype=np.float32)
    quantized_img = np.zeros_like(image, dtype=np.float32)
    zigzag_coeffs = []

    # Process each 8x8 block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]

            # Handle edge blocks by padding to 8x8
            padded_block = np.pad(block, ((0, block_size - block.shape[0]), (0, block_size - block.shape[1])), mode='constant')

            # Apply DCT, quantization, and Zigzag scan
            dct_block = apply_dct_block(padded_block)
            quantized_block = quantize_block(dct_block, quant_matrix)

            # Store DCT and quantized values in the original shape (no padding)
            dct_img[i:i + block.shape[0], j:j + block.shape[1]] = dct_block[:block.shape[0], :block.shape[1]]
            quantized_img[i:i + block.shape[0], j:j + block.shape[1]] = quantized_block[:block.shape[0], :block.shape[1]]

            # Zigzag scan for this block
            zigzag_coeffs.extend(zigzag_scan_block(quantized_block))

    return dct_img, quantized_img, zigzag_coeffs

# Apply inverse DCT to the quantized image in 8x8 blocks
def reconstruct_from_blocks(quantized_img, quant_matrix, block_size=8):
    h, w = quantized_img.shape
    reconstructed_img = np.zeros_like(quantized_img, dtype=np.float32)

    # Process each 8x8 block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = quantized_img[i:i + block_size, j:j + block_size]

            # Pad to 8x8 if necessary
            padded_block = np.pad(block, ((0, block_size - block.shape[0]), (0, block_size - block.shape[1])), mode='constant')

            # Dequantize and apply inverse DCT
            dequantized_block = dequantize_block(padded_block, quant_matrix)
            idct_block = apply_idct_block(dequantized_block)

            # Store the reconstructed block in the original shape (no padding)
            reconstructed_img[i:i + block.shape[0], j:j + block.shape[1]] = idct_block[:block.shape[0], :block.shape[1]]

    return reconstructed_img

# Function to compute Euclidean distance
def compute_euclidean_distance(original, reconstructed):
    return np.sqrt(np.sum((original - reconstructed) ** 2))

# Function to compute Mean Squared Error (MSE)
def compute_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

# Function to compute Peak Signal-to-Noise Ratio (PSNR)
def compute_psnr(original, reconstructed):
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float('inf')  # If MSE is zero, PSNR is infinite
    max_pixel_value = 255.0
    return 10 * np.log10((max_pixel_value ** 2) / mse)

# Display original and transformed images
def display_images(original, dct_image, idct_image, zigzag_coeffs, quantized_dct):
    # Sort the zigzag coefficients lexicographically
    sorted_coeffs = sorted(zigzag_coeffs)

    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(1, 5, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')

    # DCT Image
    plt.subplot(1, 5, 2)
    plt.imshow(np.log(np.abs(dct_image) + 1), cmap='gray')  # Apply log to enhance visualization
    plt.title("DCT Transformed Image")
    plt.axis('off')

    # Quantized DCT Image
    plt.subplot(1, 5, 3)
    plt.imshow(np.log(np.abs(quantized_dct) + 1), cmap='gray')
    plt.title("Quantized DCT")
    plt.axis('off')

    # IDCT (Reconstructed) Image
    plt.subplot(1, 5, 4)
    plt.imshow(idct_image, cmap='gray')
    plt.title("Reconstructed Image (IDCT)")
    plt.axis('off')

    # Zigzag Coefficients
    plt.subplot(1, 5, 5)
    plt.plot(sorted_coeffs)
    plt.title("Sorted Zigzag Coefficients")
    plt.axis('off')

    # Show the plots
    plt.show()

# Example usage
if __name__ == "__main__":
    # Input and output image paths
    image_path = "C:\\Users\\gurse\\OneDrive\\Desktop\\hackathon\\input.png"  # Replace with your image path
    output_path = "C:\\Users\\gurse\\OneDrive\\Desktop\\hackathon\\output_image.png"

    # Standard JPEG quantization matrix for an 8x8 block
    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Convert to grayscale and save it
    grayscale_image = convert_to_grayscale(image_path, output_path)
    
    # Process the image in 8x8 blocks
    dct_img, quantized_img, zigzag_coeffs = process_image_in_blocks(grayscale_image, quant_matrix)
    
    # Reconstruct the image from the quantized DCT coefficients
    idct_img = reconstruct_from_blocks(quantized_img, quant_matrix)

    # Compute metrics
    euclidean_distance = compute_euclidean_distance(grayscale_image, idct_img)
    mse = compute_mse(grayscale_image, idct_img)
    psnr = compute_psnr(grayscale_image, idct_img)
    
    # Print metrics
    print(f"Euclidean Distance: {euclidean_distance}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")

    # Display the original, DCT-transformed, quantized, IDCT-reconstructed images, and sorted Zigzag coefficients
    display_images(grayscale_image, dct_img, idct_img, zigzag_coeffs, quantized_img)
