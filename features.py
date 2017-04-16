from skimage import data
from skimage import transform as tf
from skimage.feature import CENSURE
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
import cv2

img_orig = rgb2gray(cv2.imread("catplusbrian.jpg"))


detector = CENSURE()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

detector.detect(img_orig)

ax.imshow(img_orig, cmap=plt.cm.gray)
ax.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')
ax.set_title("Original Image")

plt.tight_layout()
plt.show()
