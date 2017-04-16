import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature


image = camera()
edge_roberts = roberts(image)
edge_sobel = sobel(image)
edge_scharr = scharr(image)
edge_prewitt = prewitt(image)
edges2 = feature.canny(image, sigma=2)

fig, ax = plt.subplots(ncols=6, sharex=True, sharey=True,
                       figsize=(20, 8))

ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')

ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

ax[2].imshow(edge_scharr, cmap=plt.cm.gray)
ax[2].set_title('scharr Edge Detection')

ax[3].imshow(edge_prewitt, cmap=plt.cm.gray)
ax[3].set_title('prewitt Edge Detection')

ax[4].imshow(edges2, cmap=plt.cm.gray)
ax[4].set_title('canny Edge Detection')

ax[5].imshow(image, cmap=plt.cm.gray)
ax[5].set_title('original')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()