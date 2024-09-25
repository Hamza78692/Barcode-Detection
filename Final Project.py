import cv2
import numpy as np

def detect_barcode(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use a gradient to highlight the barcode-like regions (strong horizontal edges)
    gradientX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradientY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradientX, gradientY)
    gradient = cv2.convertScaleAbs(gradient)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.blur(gradient, (9, 9))

    # Threshold the image
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # Perform closing (dilation followed by erosion) to close gaps between barcode bars
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Perform a series of erosions and dilations to clean up the image
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours were found, return None
    if len(contours) == 0:
        return None

    # Sort the contours by area and keep the largest one, assuming it's the barcode
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw a bounding box around the detected barcode
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    # Display the image with the detected barcode
    cv2.imshow("Detected Barcode", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = 'path_to_your_image.jpg'
detect_barcode(image_path)
