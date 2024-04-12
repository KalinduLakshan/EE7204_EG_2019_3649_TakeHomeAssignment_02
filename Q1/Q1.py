import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a blank image with a gray background
image = np.full((550, 550), 128, dtype=np.uint8)

# Define pixel values for the objects
black_pixel = 0
white_pixel = 255

# Draw the original image objects
cv2.rectangle(image, (100, 100), (250, 250), black_pixel, -1)  # Black object
cv2.rectangle(image, (300, 300), (375, 375), white_pixel, -1)  # White object


# Adding Gaussian noise to the original image
def add_gaussian_noise(img, mean, sigma):
    if len(img.shape) == 2:
        h, w = img.shape  # h-height, w-width for grayscale images

        '''
        Generate Gaussian noise with a specified mean (mean) and standard deviation (sigma) 
        using NumPy's random normal function
        '''
        noise = np.random.normal(mean, sigma, (h, w))  # sigma represents the variance

        '''
        Add the noise to the original image and then clips the resulting pixel values to be within the range [0, 255]. 
        This ensures that no pixel values exceed the valid range for uint8 images.
        '''
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

    else:
        h, w, c = img.shape  # h-height, w-width, c-number of channels used for RGB images
        noise = np.random.normal(mean, sigma, (h, w, c))
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img


# Apply Gaussian noise with different parameters
noisy_image_1 = add_gaussian_noise(image, 0, 25)
noisy_image_2 = add_gaussian_noise(image, 0, 50)
noisy_image_3 = add_gaussian_noise(image, 0, 75)
noisy_image_4 = add_gaussian_noise(image, 0, 100)


# Apply Otsu's thresholding algorithm
def otsu_threshold(img):
    _, otsu_threshold_image = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_threshold_image


# Call the otsu_threshold function and assign its value to otsu_image variable
otsu_image = otsu_threshold(image)

# Plotting the images
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')

# Noisy images for different sigma values
axs[0, 1].imshow(noisy_image_1, cmap='gray')
axs[0, 1].set_title('Gaussian Noise: sigma=25')
axs[0, 2].imshow(noisy_image_2, cmap='gray')
axs[0, 2].set_title('Gaussian Noise: sigma=50')
axs[1, 0].imshow(noisy_image_3, cmap='gray')
axs[1, 0].set_title('Gaussian Noise: sigma=75')
axs[1, 1].imshow(noisy_image_4, cmap='gray')
axs[1, 1].set_title('Gaussian Noise: sigma=100')

# Otsu's image
axs[1, 2].imshow(otsu_image, cmap='gray')
axs[1, 2].set_title("Otsu's Algorithm")

plt.show()
