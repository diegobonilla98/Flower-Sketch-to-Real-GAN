import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.io import loadmat


labels = loadmat('/media/bonilla/HDD_2TB_basura/databases/102flowers/imagelabels.mat')['labels'][0]
# labels = np.load('/media/bonilla/HDD_2TB_basura/databases/102flowers/flower_labels.npy')
print()

# images = glob.glob('/media/bonilla/HDD_2TB_basura/databases/102flowers/jpg/*')
# for image_path in images:
#     image_org = cv2.imread(image_path)[:, :, ::-1]
#     image = cv2.resize(image_org, (224, 224))
#     image = vgg16.preprocess_input(image)
#     image = np.expand_dims(image, axis=0)
#     yhat = model.predict(image)
#     labels = vgg16.decode_predictions(yhat)
#
#     print(labels)
#     plt.imshow(image_org)
#     plt.show()
