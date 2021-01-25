import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from random import choice
import tqdm


def to_sketch_method_A(image):
    k_size = 2 * np.random.randint(6, 10) + 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (k_size, k_size), 0)
    return 255 - cv2.Canny(blur, 100, 150)


def to_sketch_method_B(image):
    kernel = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], np.uint8)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_dilated = cv2.dilate(img_gray, kernel, iterations=1)
    img_diff = cv2.absdiff(img_dilated, img_gray)
    return 255 - img_diff


def get_mask(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = 255 - cv2.inRange(hsv, (117, 200, 120), (145, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return mask


def is_valid(mask, threshold=20):
    h, w = mask.shape
    per = cv2.countNonZero(m) * 100 / (h * w)
    return per > threshold


images = glob.glob('/media/bonilla/HDD_2TB_basura/databases/102flowers/102segmentations/segmim/*')
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
for image in images:
    m = get_mask(image)

    i = cv2.imread(image)

    rand = np.random.rand(5)
    if rand[0] > 0.5:
        i = cv2.flip(i, 0)
        m = cv2.flip(m, 0)
    if rand[1] > 0.5:
        i = cv2.flip(i, 1)
        m = cv2.flip(m, 1)
    if rand[2] > 0.5:
        rot = np.random.randint(0, 3)
        i = cv2.rotate(i, rot)
        m = cv2.rotate(m, rot)
    if rand[3] > 0.5:
        l, a, b = cv2.split(cv2.cvtColor(i, cv2.COLOR_BGR2LAB))
        l = clahe.apply(l)
        i = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    if rand[4] > 0.5:
        gamma = np.random.rand() * 3.
        invGamma = 1.0 / gamma
        table = np.array([((ii / 255.0) ** invGamma) * 255
                          for ii in np.arange(0, 256)]).astype("uint8")
        i = cv2.LUT(i, table)

    i = np.uint8(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) / 255. * i)

    plt.figure(0)
    plt.imshow(i[:, :, ::-1])

    plt.show()


