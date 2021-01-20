import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


class DataLoader:
    def __init__(self, img_size):
        self.ROOT = '/media/bonilla/HDD_2TB_basura/databases/all_flowers'
        self.IMAGES = np.array(glob.glob(os.path.join(self.ROOT, '*')))
        self.NUM_IMAGES = len(self.IMAGES)
        self.IMG_SIZE = img_size

    @staticmethod
    def to_sketch(image):
        k_size = 2 * np.random.randint(1, 6) + 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (k_size, k_size), 0)
        return 255 - cv2.Canny(blur, 10, 80)

    def load_image(self, path, label: str):
        image = cv2.imread(path)
        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        if label.lower() == 'x':
            image = self.to_sketch(image)
            image = np.expand_dims(image, axis=-1)
        elif label.lower() == 'y':
            image = image
        return (image.astype('float32') - 127.5) / 127.5

    def load_batch(self, batch_size):
        paths = np.random.choice(self.IMAGES, size=(batch_size, ), replace=False)
        X = np.array([self.load_image(p, label='x') for p in paths])
        Y = np.array([self.load_image(p, label='y') for p in paths])
        return X, Y
