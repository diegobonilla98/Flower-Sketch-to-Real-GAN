import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob
from tensorflow.keras.models import load_model
from BonillaGAN import GANSR
from DataLoader import DataLoader
from tensorflow.keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


data_loader = DataLoader(256)
asr = GANSR(data_loader)
model = asr.get_generator(8000)

sketches = glob.glob('./test_sketches/*')
for i, sk in enumerate(sketches):
    image_org = cv2.imread(sk)
    image_org = cv2.resize(image_org, (256, 256))
    image_sk = data_loader.combine_sketches(image_org)
    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    image = cv2.addWeighted(image, 0.5, image_sk, 0.5, 1.0)
    image = np.expand_dims(np.expand_dims((image.astype('float32') - 127.5) / 127.5, axis=-1), axis=0)
    res = model.predict(image)

    plt.figure(figsize=(15, 15))
    plt.imshow((np.hstack([(image_org - 127.5) / 127.5, res[0, :, :, ::-1]]) + 1) / 2)
    plt.axis('off')
    plt.savefig(f'test_res_{i}.png')
    plt.show()
