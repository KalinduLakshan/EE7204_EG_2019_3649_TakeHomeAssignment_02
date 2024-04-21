import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def add_gaussian_noise(img, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy_img = cv.add(img.astype(np.float32), gauss)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def apply_otsus_threshold(img):
    """Applies Otsu's thresholding to an image."""
    _, thresh_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh_img


# Create an image with a black background
height, width = 200, 200
image = np.zeros((height, width), dtype=np.uint8)

# Draw a gray rectangle
cv.rectangle(image, (50, 50), (150, 120), 100, -1)  # Gray value

# Draw a white square
cv.rectangle(image, (70, 130), (130, 190), 200, -1)  # White value

# Generate a noisy version of the image
noisy_image = add_gaussian_noise(image)

# Apply Otsu's thresholding
thresholded_image = apply_otsus_threshold(noisy_image)

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Noisy Image')
axes[1].axis('off')

axes[2].imshow(thresholded_image, cmap='gray')
axes[2].set_title('Thresholded Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()
