import cv2
import numpy as np


def get_eight_neighbour(x, y, shape):
    # Returns the coordinates of the 8-neighbors of a point within a given shape
    output = []
    max_x = shape[1] - 1
    max_y = shape[0] - 1

    # Define the eight possible neighbors and constrain them within the image boundaries.
    neighbors = [
        (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),  # Top row
        (x - 1, y),                     (x + 1, y),   # Middle row (left and right)
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)  # Bottom row
    ]

    for nx, ny in neighbors:
        # Constrain the coordinates within the image boundaries
        output_x = min(max(nx, 0), max_x)
        output_y = min(max(ny, 0), max_y)
        output.append((output_x, output_y))

    return output


def region_growing(im, seed):
    # Performs region growing from a given seed point
    seed_points = [seed]  # Start with the initial seed point
    output_img = np.zeros_like(img)  # Output image initialized to zeros (black)
    processed = set()  # A set to track processed points

    while seed_points:
        pix = seed_points.pop(0)  # Get the next seed point to process
        if pix in processed:
            continue  # If already processed, skip

        output_img[pix[0], pix[1]] = 255  # Mark the point as white in the output image
        processed.add(pix)  # Add to the set of processed points

        # Add neighbors to the seed points if they haven't been processed
        for coord in get_eight_neighbour(pix[0], pix[1], img.shape):
            if img[coord[0], coord[1]] == 255 and coord not in processed:
                seed_points.append(coord)

    return output_img


# Define the predefined seed point
predefined_seed = (100, 10)  # (y, x) coordinate for the seed point

# Load the image and convert to grayscale
image = cv2.imread('apple.jpg', 0)

# Apply a binary threshold to create a binary image
ret, img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Use the predefined seed point for region growing
rg_out = region_growing(img, predefined_seed)

# Display the original and the region-grown images
cv2.imshow('Original Image', image)
cv2.imshow('Region Growing Image', rg_out)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Close all OpenCV windows
