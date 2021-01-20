import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from random import choice

images = glob.glob('/media/bonilla/HDD_2TB_basura/databases/all_flowers/*')
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

image = cv2.imread(choice(images))

success, saliencyMap = saliency.computeSaliency(image)
# saliencyMap = (saliencyMap * 255).astype("uint8")


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
contour = 255 - img_diff

res = np.uint8(saliencyMap * contour)

plt.figure(0)
plt.imshow(image[:, :, ::-1])

plt.figure(1)
plt.imshow(saliencyMap, cmap='gray')

plt.figure(2)
plt.imshow(contour, cmap='gray')

plt.figure(3)
plt.imshow(res, cmap='gray')

plt.show()
