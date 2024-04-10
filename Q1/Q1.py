# Import the necessary packages
import cv2
import numpy as np

# Create a blank image
image = np.zeros((550, 550), dtype=np.uint8)

# Define pixel values for each component of the image
background_pixel = 0
object1_pixel = 128
object2_pixel = (300, 255, 255)

# Draw the objects where the background is selected to be black
cv2.rectangle(image, (100, 100), (250, 250), object1_pixel, -1)  # Object 1
cv2.rectangle(image, (300, 300), (375, 375), object2_pixel, -1)  # Object 2

# Save the image
cv2.imwrite("Image with 2 objects & total of 3 pixel values.png", image)

# Display the image
#cv2.imshow("Image with 2 objects & total of 3 pixel values", image)

# Adding Gaussian noise to the image
"""
    Args:
    - img: Input image (numpy array).
    - mean: Mean of the Gaussian distribution.
    - sigma: Standard deviation of the Gaussian distribution.

    Returns:
    - Noisy image (numpy array).
"""


def add_gaussian_noise(img, mean, sigma):
    # For a Grayscale image
    if len(img.shape) == 2:
        h, w = img.shape  # h-height, w-width of the image
        noise = np.random.normal(mean, sigma, (h, w))
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    else:
        # For an RGB image
        h, w, c = img.shape  # h-height, w-width, c-number of channels of the image
        noise = np.random.normal(mean, sigma, (h, w, c))
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img


# Call the add_gaussian_noise function
# For mean=0, sigma=25
noisy_image_1 = add_gaussian_noise(image, 0, 25)
# For mean=0, sigma=50
noisy_image_2 = add_gaussian_noise(image, 0, 50)
# For mean=0, sigma=75
noisy_image_3 = add_gaussian_noise(image, 0, 75)
# For mean=0, sigma=100
noisy_image_4 = add_gaussian_noise(image, 0, 100)

'''
# Image after adding the Gaussian Noise
cv2.imshow("Gaussian Noise added image with mean=0 & sigma=25", noisy_image_1)
cv2.imshow("Gaussian Noise added image with mean=0 & sigma=50", noisy_image_2)
cv2.imshow("Gaussian Noise added image with mean=0 & sigma=75", noisy_image_3)
cv2.imshow("Gaussian Noise added image with mean=0 & sigma=100", noisy_image_4)
'''


# Applying the Otsu's algorithm to te input image for image segmentation
def otsu_threshold(img):
    """
        Args:
        - img: Input image (numpy array).

        Returns:
        - Binary image after thresholding (numpy array).
    """
    _, otsu_threshold_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_threshold_image


img1 = cv2.imread('malki.jpg', cv2.IMREAD_GRAYSCALE)
otsu_image = otsu_threshold(img1)

# Image after applying the Otsu's Algorithm
cv2.imshow("Image with Otsu'slgorithm", otsu_image)

cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()  # Close all OpenCV windows
