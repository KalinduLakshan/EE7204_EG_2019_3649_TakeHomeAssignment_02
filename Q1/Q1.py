import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define image dimensions
height, width = 100, 100

# Create a blank image with a background and two objects
image = np.zeros((height, width), dtype=np.uint8)

# Define pixel values for the objects and background
object1_pixel = 50
object2_pixel = 200
background_pixel = 255

# Draw the objects
cv2.rectangle(image, (20, 20), (40, 40), object1_pixel, -1)  # Object 1
cv2.rectangle(image, (60, 60), (80, 80), object2_pixel, -1)  # Object 2

# Display the original image
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()


# Adding Gaussian noise to the original image
def add_gaussian_noise(img, mean, sigma):
    h, w = img.shape  # Image dimensions
    noise = np.random.normal(mean, sigma, (h, w))  # Generate Gaussian noise
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)  # Add noise to the image
    return noisy_img


# Apply Gaussian noise with mean=0 and different sigma values
noisy_image_sigma_10 = add_gaussian_noise(image, 0, 10)
noisy_image_sigma_20 = add_gaussian_noise(image, 0, 20)


# Define Otsu's thresholding algorithm
def otsu_threshold(img):
    _, otsu_threshold_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_threshold_image


# Apply Otsu's algorithm to noisy images with sigma=10
otsu_image_sigma_10 = otsu_threshold(noisy_image_sigma_10)
otsu_image_sigma_20 = otsu_threshold(noisy_image_sigma_20)

# Plotting the images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

# Noisy image with sigma=10
axs[1].imshow(noisy_image_sigma_10, cmap='gray')
axs[1].set_title('Noisy Image (sigma=10)')
axs[1].axis('off')

# Otsu's image with sigma=10
axs[2].imshow(otsu_image_sigma_10, cmap='gray')
axs[2].set_title("Otsu's Algorithm (sigma=10)")
axs[2].axis('off')

# Show the plots
plt.tight_layout()
plt.show()
