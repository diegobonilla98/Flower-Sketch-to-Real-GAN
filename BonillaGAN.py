import cv2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Conv2DTranspose, Input, UpSampling2D, LeakyReLU, \
    PReLU, add, Dropout, BatchNormalization, Lambda, Activation, Dense, Flatten, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import vgg16
import tensorflow.keras.backend as K
from DataLoader import DataLoader
import tensorflow as tf
from tensorflow.keras.losses import Huber, MAE, MSE
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


def perceptual_loss(img_true, img_generated):
    img_true *= 127.5
    img_generated *= 127.5
    full_vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    loss_block3 = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block3_conv3').output)
    loss_block3.trainable = False
    loss_block2 = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block2_conv2').output)
    loss_block2.trainable = False
    loss_block1 = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block1_conv2').output)
    loss_block1.trainable = False
    return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2 * K.mean(
        K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5 * K.mean(
        K.square(loss_block3(img_true) - loss_block3(img_generated)))


def mixed_loss(y_true, y_pred):
    mae = MAE(y_true, y_pred)
    return mae * 10 + perceptual_loss(y_true, y_pred) / 100_000


def set_trainable(m, val):
    m.trainable = val
    for l in m.layers:
        l.trainable = val


class GANSR:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.gf = 128
        self.channels = 3
        self.image_shape = (256, 256, 1)
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.init = RandomNormal(mean=0.0, stddev=0.02)

        self.generator = self.build_generator()
        self.generator.summary()
        plot_model(self.generator, to_file='generator_model.png')

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', loss_weights=[0.5], optimizer=self.optimizer, metrics=['acc'])
        self.discriminator.summary()
        plot_model(self.discriminator, to_file='discriminator_model.png')

        set_trainable(self.discriminator, False)
        input_tensor = Input(shape=self.image_shape)
        gen = self.generator(input_tensor)
        dis = self.discriminator([input_tensor, gen])
        self.adversarial = Model(input_tensor, [dis, gen])
        self.adversarial.summary()
        # 5e-5
        self.adversarial.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1., 100.],
                                 optimizer=self.optimizer)
        plot_model(self.adversarial, to_file='adversarial_model.png')

    def build_discriminator(self):
        filters = 64
        input_tensor_A = Input(shape=(self.image_shape[0], self.image_shape[1], 1))
        input_tensor_B = Input(shape=(self.image_shape[0], self.image_shape[1], 3))

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same', kernel_initializer=self.init)(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization()(d)
                # d = Dropout(0.3)(d)
            return d

        input_tensor = Concatenate()([input_tensor_A, input_tensor_B])
        x = conv2d_block(input_tensor, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters * 2)
        x = conv2d_block(x, filters * 2, strides=2)
        x = conv2d_block(x, filters * 4)
        x = conv2d_block(x, filters * 4, strides=2)
        x = conv2d_block(x, filters * 8)
        x = conv2d_block(x, filters * 8, strides=2)

        x = Dense(filters * 16, kernel_initializer=self.init)(x)
        x = LeakyReLU(alpha=0.2)(x)
        output = Dense(1, activation='sigmoid', kernel_initializer=self.init)(x)

        model = Model([input_tensor_A, input_tensor_B], output)

        return model

    def build_generator(self):
        def conv2d(layer_input, filters=16, strides=1, name=None, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name, kernel_initializer=self.init)(layer_input)
            d = BatchNormalization(name=name + "_bn")(d)
            d = PReLU(shared_axes=[1, 2])(d)
            d = Dropout(0.3)(d)
            return d

        def residual(layer_input, filters=16, strides=1, name=None, f_size=3):
            d = conv2d(layer_input, filters=filters, strides=strides, name=name, f_size=f_size)
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name + "_2", kernel_initializer=self.init)(d)
            d = BatchNormalization(name=name + "_bn2")(d)
            d = add([d, layer_input])
            return d

        def conv2d_transpose(layer_input, filters=16, strides=1, name=None, f_size=4):
            u = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
            u = UpSampling2D(size=2, interpolation='bilinear')(u)
            u = BatchNormalization(name=name + "_bn")(u)
            u = PReLU(shared_axes=[1, 2])(u)
            u = Dropout(0.3)(u)
            return u

        input_tensor = Input(shape=self.image_shape)
        c1 = conv2d(input_tensor, filters=self.gf, strides=1, name="g_e1", f_size=7)
        c2 = conv2d(c1, filters=self.gf * 2, strides=2, name="g_e2", f_size=3)
        c3 = conv2d(c2, filters=self.gf * 4, strides=2, name="g_e3", f_size=3)

        r1 = residual(c3, filters=self.gf * 4, name='g_r1')
        r2 = residual(r1, self.gf * 4, name='g_r2')
        r3 = residual(r2, self.gf * 4, name='g_r3')
        r4 = residual(r3, self.gf * 4, name='g_r4')
        r5 = residual(r4, self.gf * 4, name='g_r5')
        r6 = residual(r5, self.gf * 4, name='g_r6')
        r7 = residual(r6, self.gf * 4, name='g_r7')
        r8 = residual(r7, self.gf * 4, name='g_r8')
        r9 = residual(r8, self.gf * 4, name='g_r9')

        d1 = conv2d_transpose(r9, filters=self.gf * 2, f_size=3, strides=2, name='g_d1_dc')
        d2 = conv2d_transpose(d1, filters=self.gf, f_size=3, strides=2, name='g_d2_dc')

        output_img = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh', kernel_initializer=self.init)(d2)

        return Model(inputs=input_tensor, outputs=output_img)

    def plot_images(self, epoch):
        x, y = self.data_loader.load_batch(batch_size=3)
        res = self.generator.predict(x)
        comb = (np.vstack([np.hstack([y[0], cv2.cvtColor(x[0], cv2.COLOR_GRAY2BGR), res[0]]),
                           np.hstack([y[1], cv2.cvtColor(x[1], cv2.COLOR_GRAY2BGR), res[1]]),
                           np.hstack([y[2], cv2.cvtColor(x[2], cv2.COLOR_GRAY2BGR), res[2]])]) + 1) / 2
        plt.figure(figsize=(15, 15))
        plt.imshow(comb[:, :, ::-1])
        plt.axis('off')
        plt.savefig(f'./results/epoch_{epoch}.jpg')
        plt.close()

    def resume_from(self, epoch):
        self.generator = load_model(f'/media/bonilla/HDD_2TB_basura/models/Flower_Sketch_to_Real/gen_epoch_{epoch}.h5')
        self.discriminator = load_model(f'/media/bonilla/HDD_2TB_basura/models/Flower_Sketch_to_Real/dis_epoch_{epoch}.h5')
        self.adversarial = load_model(f'/media/bonilla/HDD_2TB_basura/models/Flower_Sketch_to_Real/adv_epoch_{epoch}.h5')

    def get_generator(self, epoch=None):
        if epoch is not None:
            self.generator = load_model(
                f'/media/bonilla/HDD_2TB_basura/models/Flower_Sketch_to_Real/gen_epoch_{epoch}.h5')
        return self.generator

    def fit(self, epochs, batch_size):
        set_trainable(self.generator, True)
        for epoch in range(epochs):
            real_X, real_Y = self.data_loader.load_batch(batch_size=batch_size)
            real = np.ones((batch_size, 16, 16, 1))
            # real_y = np.random.uniform(0.8, 1., size=(batch_size, 16, 16, 1))

            fake_Y = self.generator.predict(real_X)
            fake = np.zeros((batch_size, 16, 16, 1))
            # fake_y = np.random.uniform(0., 0.2, size=(batch_size, 16, 16, 1))

            set_trainable(self.discriminator, True)
            if np.random.rand() <= 0.1:
                real, fake = fake, real
            if 3000 > epoch > 2000 and np.random.rand() <= 0.55:
                real, fake = fake, real
            real_X += np.random.normal(0., 0.5, size=self.image_shape)
            d_loss_true = self.discriminator.train_on_batch([real_X, real_Y], real)
            d_loss_fake = self.discriminator.train_on_batch([real_X, fake_Y], fake)

            # real_y = np.random.uniform(0.8, 1., size=(batch_size, 16, 16, 1))
            set_trainable(self.discriminator, False)
            real = np.ones((batch_size, 16, 16, 1))
            g_loss = self.adversarial.train_on_batch(real_X, [real, real_Y])

            print(f"[Epoch: {epoch}/{epochs}]\t[adv_loss: {g_loss}, d_fake: {d_loss_fake}, d_true: {d_loss_true}]")

            if epoch % 25 == 0:
                self.plot_images(epoch)
                if epoch % 100 == 0:
                    self.generator.save(f'/media/bonilla/HDD_2TB_basura/models/Flower_Sketch_to_Real/gen_epoch_{epoch}.h5')
                    self.discriminator.save(f'/media/bonilla/HDD_2TB_basura/models/Flower_Sketch_to_Real/dis_epoch_{epoch}.h5')
                    self.adversarial.save(f'/media/bonilla/HDD_2TB_basura/models/Flower_Sketch_to_Real/adv_epoch_{epoch}.h5')
            if epoch > 500:
                self.optimizer.learning_rate.assign(0.0002 * (0.43 ** epoch))


if __name__ == '__main__':
    data_loader = DataLoader(256)
    asr = GANSR(data_loader)
    asr.fit(10_000, 3)
