import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob
from BonillaGAN import GANSR
from DataLoader import DataLoader
from tensorflow.keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


data_loader = DataLoader(256)
asr = GANSR(data_loader, 3)
model = asr.get_generator(9900)

sketches = glob.glob('./test_sketches/*')
for i, sk in enumerate(sketches):
    image_org = cv2.imread(sk)
    image_org = cv2.resize(image_org, (256, 256), cv2.INTER_LANCZOS4)
    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)

    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 27)
    image = data_loader.combine_sketches(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = np.expand_dims((image.astype('float32') - 127.5) / 127.5, axis=0)
    res = model.predict(image)

    plt.figure(figsize=(15, 7))
    plt.imshow((np.hstack([(image_org - 127.5) / 127.5, res[0, :, :, ::-1]]) + 1) / 2)
    plt.axis('off')
    plt.savefig(f'test_res_{i}.png')
    plt.show()
