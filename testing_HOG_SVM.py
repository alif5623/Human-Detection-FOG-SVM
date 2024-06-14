from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from joblib import load 
from joblib import dump  
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
import time

# Define HOG Parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# define the sliding window:
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

# Upload the saved svm model:
model = load('svm-hog-human.npy')

# Test the trained classifier on an image below!
scale = 0
detections = []

# Read the image you want to detect the object in:
img = cv2.imread("360_F_286933709_B08Z0FymzypXREDHB8YkCD7UgNakqKOs.jpg")

# Resize the image if necessary
img = cv2.resize(img, (1024, 1024)) # can change the size to default by commenting this code out or putting in a random number

# Define the size of the sliding window
(winW, winH) = (1024, 1024)  # Adjusted window size to be more typical
windowSize = (winW, winH)
downscale = 1.5

# Start the timer
start_time = time.time()

# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=1.5):
    for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        window = color.rgb2gray(window)
        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')
        fds = fds.reshape(1, -1)
        pred = model.predict(fds)
        
        if pred == 1:
            if model.decision_function(fds) > 0.6:
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale, model.decision_function(fds)))
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0] * (downscale**scale)),
                                   int(windowSize[1] * (downscale**scale))))
    scale += 1

# Calculate the time taken and FPS
end_time = time.time()
time_taken = end_time - start_time
fps = 1 / time_taken

print(f"Time taken: {time_taken} seconds")
print(f"FPS: {fps}")

# Draw detections
clone = resized.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness=2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
sc = np.array(sc)
pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("Raw Detections after NMS", img)

k = cv2.waitKey(0) & 0xFF 
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('Path/to_the_directory/of_saved_image.png', img)
    cv2.destroyAllWindows()
