import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data, exposure
from skimage.segmentation import active_contour
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw
from skimage.restoration import inpaint


def init_size(x_radius, y_radius, x_shift, y_shift):
    s = np.linspace(0, 2*np.pi, 400)
    x = x_shift + x_radius*np.cos(s)
    y = y_shift + y_radius*np.sin(s)
    return np.array([x, y]).T


def cross(o, a, b):
    """ 2D cross product of OA and OB vectors,
     i.e. z-component of their 3D cross product.
    :param o: point O
    :param a: point A
    :param b: point B
    :return cross product of vectors OA and OB (OA x OB),
     positive if OAB makes a counter-clockwise turn,
     negative for clockwise turn, and zero
     if the points are colinear.
    """

    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def inside_hull(hull_vertices, point):

    inside = True
    for ind in range(1, len(hull_vertices)):
        res = cross(hull_vertices[ind - 1], hull_vertices[ind], point)
        if res < 0:
            inside = False

    return inside


def np_array_to_image(arr):
    arr_8 = (((arr - arr.min()) / (arr.max() - arr.min())) * 255.9).astype(np.uint8)

    res_img = Image.fromarray(arr_8)
    return res_img


def remove_object(img, bounding_region, im_name, params={}):
    """img, bounding_region, name"""
    x_width, y_width = img.shape
    dims = y_width, x_width

    x_rad = bounding_region[0]
    y_rad = bounding_region[1]
    x_pos = bounding_region[2]
    y_pos = bounding_region[3]

    alpha = params.get("alpha", 0.015)
    beta = params.get("beta", 7)
    gamma = params.get("gamma", 0.001)
    w_edge = params.get("w_edge", 1)
    w_line = params.get("w_line", 0)
    bc = params.get("bc", 'periodic')
    max_px_move = params.get("max_px_move", 1.0)
    max_iterations = params.get("max_iterations", 2500)
    convergence = params.get("convergence", 0.1)

    # find the actual shape
    init = init_size(x_rad, y_rad, x_pos, y_pos)
    snake = active_contour(exposure.equalize_hist(img), init, alpha=alpha,
                           beta=beta, gamma=gamma,
                           w_edge=w_edge, w_line=w_line,
                           bc=bc, max_px_move=max_px_move,
                           max_iterations=max_iterations, convergence=convergence)

    snake = np.append(snake, [snake[0]], axis=0)

    mask = Image.new('RGB', dims)
    drw = ImageDraw.Draw(mask, 'RGBA')
    drw.polygon(list(map(tuple, snake)), (255, 255, 255, 255))
    del drw

    mask.save("output/" + im_name + '-mask-out.png', 'PNG')
    mask_np = rgb2gray(np.asarray(mask, dtype='uint8'))

    image_result = inpaint.inpaint_biharmonic(img, mask_np, multichannel=False)

    q_img = np_array_to_image(image_result)
    q_img.save("output/" + im_name + "-inpaint-out.png")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(image_result, shape=dims)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, image_result.shape[1], image_result.shape[0], 0])
    plt.show()


images = (
            data.astronaut(),
            data.camera(),
            data.chelsea(),
            data.coffee(),
            data.coins()
        )


image_tags = (
                "astronaut",
                "camera",
                "chelsea",
                "coffee",
                "coins"
            )

image_pos = (
                [100, 100, 220, 100],
                [35, 70, 394, 267],
                [30, 30, 314, 128],
                [60, 60, 364, 285],
                [30, 30, 102, 198]
            )


for idx, name in enumerate(image_tags):
    start = time.time()
    remove_object(rgb2gray(images[idx]), image_pos[idx], name)
    end = time.time()
    print(idx, ":", name, "time elapsed", end - start)
    q_img = np_array_to_image(images[idx])
    q_img.save("output/" + name + "-original.png", 'PNG')


images = (
            cv2.imread("pidgeon.jpg"),
            cv2.imread("14877-diminish_reality_teaser.jpg"),
            cv2.imread("object-removal-before.jpg"),
            cv2.imread("output/new-coin-original.png"),
            cv2.imread("new-boat.png")
        )


image_tags = (
                "pidgeon",
                "usb",
                "boat",
                "new-coin",
                "new-boat"
            )

image_pos = (
                [30, 30, 192, 138],
                [35, 45, 127, 94],
                [150, 50, 140, 250],
                [20, 20, 96, 182],
                [50, 50, 160, 280]
            )


for idx, name in enumerate(image_tags):
    start = time.time()
    remove_object(rgb2gray(images[idx]), image_pos[idx], name)
    end = time.time()
    q_img = np_array_to_image(images[idx])
    q_img.save("output/" + name + "-original.png", 'PNG')
    print(idx, ":", name, "time elapsed", end - start)

