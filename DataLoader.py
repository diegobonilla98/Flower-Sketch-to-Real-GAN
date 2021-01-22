import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


class DataLoader:
    def __init__(self, img_size):
        self.ROOT = '/media/bonilla/HDD_2TB_basura/databases/102flowers/102segmentations/segmim'
        self.IMAGES = np.array(glob.glob(os.path.join(self.ROOT, '*')))
        self.NUM_IMAGES = len(self.IMAGES)
        self.IMG_SIZE = img_size
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    @staticmethod
    def to_sketch_method_A(image):
        k_size = 2 * np.random.randint(1, 6) + 1
        init = np.random.randint(10, 100)
        diff = np.random.randint(10, 100)
        if len(image.shape) == 3 or image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (k_size, k_size), 0)
        return 255 - cv2.Canny(blur, init, init + diff)

    @staticmethod
    def to_sketch_method_B(image):
        kernel = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ], np.uint8)
        if len(image.shape) == 3 or image.shape[-1] == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image
        img_dilated = cv2.dilate(img_gray, kernel, iterations=1)
        img_diff = cv2.absdiff(img_dilated, img_gray)
        return 255 - img_diff

    def get_mask(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = 255 - cv2.inRange(hsv, (117, 200, 120), (145, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        return mask, image

    def augment(self, i, m, rand, rot, gamma):
        if rand[0] > 0.5:
            i = cv2.flip(i, 0)
            m = cv2.flip(m, 0)
        if rand[1] > 0.5:
            i = cv2.flip(i, 1)
            m = cv2.flip(m, 1)
        if rand[2] > 0.5:
            i = cv2.rotate(i, rot)
            m = cv2.rotate(m, rot)
        if rand[3] > 0.5:
            l, a, b = cv2.split(cv2.cvtColor(i, cv2.COLOR_BGR2LAB))
            l = self.clahe.apply(l)
            i = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        if rand[4] > 0.5:
            invGamma = 1.0 / gamma
            table = np.array([((ii / 255.0) ** invGamma) * 255 for ii in np.arange(0, 256)]).astype("uint8")
            i = cv2.LUT(i, table)
        return i, m

    def combine_sketches(self, image):
        iA = self.to_sketch_method_A(image)
        iB = self.to_sketch_method_B(image)
        per = np.random.uniform(0.3, 0.7)
        return np.uint8(iA * per + iB * (1 - per))

    def load_image(self, path, label: str, rand, rot, gamma):
        mask, image = self.get_mask(path)
        image_aug, mask_aug = self.augment(image, mask, rand, rot, gamma)
        cropped = np.where(cv2.cvtColor(mask_aug, cv2.COLOR_GRAY2BGR) == (255, 255, 255),
                           image_aug, (255, 255, 255)).astype(np.uint8)
        if label.lower() == 'x':
            image = self.combine_sketches(cropped)
            image = np.expand_dims(image, axis=-1)
        elif label.lower() == 'y':
            image = cropped
        return (image.astype('float32') - 127.5) / 127.5

    def load_batch(self, batch_size):
        paths = np.random.choice(self.IMAGES, size=(batch_size, ), replace=False)
        rand = np.random.rand(5)
        rot = np.random.randint(0, 3)
        gamma = np.random.rand() * 2. + 0.5
        X = np.array([self.load_image(p, 'x', rand, rot, gamma) for p in paths])
        Y = np.array([self.load_image(p, 'y', rand, rot, gamma) for p in paths])
        return X, Y
