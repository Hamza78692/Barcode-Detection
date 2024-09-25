import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Convert the image to grayscale
def rgb2gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

# Apply Sobel filter for edge detection
def sobel_filter(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = convolve(image, Kx)
    Iy = convolve(image, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    return G

# Convolution operation
def convolve(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Output image
    output = np.zeros(image.shape)

    # Convolution operation
    for x in range(image_w):
        for y in range(image_h):
            output[y, x] = np.sum(kernel * padded_image[y: y + kernel_h, x: x + kernel_w])

    return output

# Simple barcode detection
def detect_barcode(image):
    gray_image = rgb2gray(image)
    edges = sobel_filter(gray_image)
    
    # Average the rows to find the barcode pattern (vertical lines)
    row_sums = np.mean(edges, axis=1)

    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')

    plt.subplot(1, 2, 2)
    plt.plot(row_sums)
    plt.title('Detected Barcode Pattern')
    plt.show()

# Load the image
image = np.array(Image.open('barcode_image.png'))

# Detect the barcode
detect_barcode(image)
