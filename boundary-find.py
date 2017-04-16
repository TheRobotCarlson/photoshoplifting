import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2
import numpy as np
from skimage.feature import CENSURE
import time
from PIL import Image, ImageDraw
from skimage.restoration import inpaint
from skimage.segmentation import find_boundaries


img = rgb2gray(data.astronaut())


test = find_boundaries(img, mode='inner')

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(test, shape=test.shape)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, test.shape[1], test.shape[0], 0])
plt.show()
