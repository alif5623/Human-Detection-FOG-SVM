import cv2
from skimage import io
from skimage.feature import hog
from skimage import color, exposure
import numpy as np

# Load the image using skimage.io (or directly with cv2.imread if preferred)
img = io.imread(r"human detection dataset\1\1.png")
img = cv2.resize(img, (512, 512))  # Resize if necessary

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Calculate HOG features
fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

# Rescale HOG image intensities for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

# Display the images using OpenCV (cv2)
cv2.imshow('Input image', gray)
cv2.imshow('Histogram of Oriented Gradients', hog_image_rescaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
