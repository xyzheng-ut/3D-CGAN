import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


def pad_dim(x, n=1):
    x = tf.concat((x[:, :, :, -n:, :], x, x[:, :, :, :n, :]), axis=-2)
    x = tf.concat((x[:, :, -n:, :, :], x, x[:, :, :n, :, :]), axis=-3)
    x = tf.concat((x[:, -n:, :, :, :], x, x[:, :n, :, :, :]), axis=-4)
    return x


def delete_dim(x):
    x = x[:, 3:-3, 3:-3, 3:-3, :]
    return x


class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.fc = layers.Dense(2 * 2 * 2 * 512)
        self.bn0 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv1 = layers.Conv3DTranspose(256, kernel_size=[4, 4, 4], strides=[2, 2, 2], padding='valid')
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv2 = layers.Conv3DTranspose(128, kernel_size=4, strides=2, padding='valid')
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv3 = layers.Conv3DTranspose(64, kernel_size=4, strides=2, padding='valid')
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv4 = layers.Conv3DTranspose(32, kernel_size=4, strides=2, padding='valid')
        self.bn4 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv5 = layers.Conv3DTranspose(16, kernel_size=4, strides=2, padding='valid')
        self.bn5 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv6 = layers.Conv3DTranspose(1, kernel_size=4, strides=2, padding='valid')

    def call(self, inputs_noise, input_condition, training=None):
        # inputs_noise: (b, 128), inputs_condition: (b, 2)
        inputs_noise = tf.cast(inputs_noise, dtype=tf.float32)
        input_condition = tf.cast(input_condition, dtype=tf.float32)
        net = tf.concat((inputs_noise, input_condition), axis=-1)
        net = self.fc(net)  # (b, 4*4*512)
        net = self.bn0(net, training=training)

        net = tf.reshape(net, [-1, 2, 2, 2, 512])  # (b, 4, 4, 512)

        net = pad_dim(net, n=1)
        net = tf.nn.leaky_relu(self.bn1(self.conv1(net), training=training))
        net = delete_dim(net)

        net = pad_dim(net)
        net = tf.nn.leaky_relu(self.bn2(self.conv2(net), training=training))
        net = delete_dim(net)

        net = pad_dim(net)
        net = tf.nn.leaky_relu(self.bn3(self.conv3(net), training=training))
        net = delete_dim(net)

        net = pad_dim(net)
        net = tf.nn.leaky_relu(self.bn4(self.conv4(net), training=training))
        net = delete_dim(net)

        # net = pad_dim(net)
        # net = tf.nn.leaky_relu(self.bn5(self.conv5(net), training=training))
        # net = delete_dim(net)

        net = pad_dim(net)
        net = self.conv6(net)
        net = delete_dim(net)

        net = tf.sigmoid(net)  # (b, 256, 256, 1)

        return net


class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv3D(16, kernel_size=4, strides=2, padding='valid')  # => (b, 128, 128, 16)
        self.conv2 = layers.Conv3D(32, kernel_size=4, strides=2, padding='valid')  # => (b, 64, 64, 32)
        self.conv3 = layers.Conv3D(64, kernel_size=4, strides=2, padding='valid')  # => (b, 32, 32, 64)
        self.conv4 = layers.Conv3D(128, kernel_size=4, strides=2, padding='valid')  # => (b, 16, 16, 128)
        self.conv5 = layers.Conv3D(256, kernel_size=4, strides=2, padding='valid')  # => (b, 8, 8, 256)
        self.conv6 = layers.Conv3D(512, kernel_size=4, strides=2, padding='valid')  # => (b, 2, 2, 512)

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs_img, training=None):
        inputs_img = tf.cast(inputs_img, dtype=tf.float32)
        x = pad_dim(inputs_img)

        # inputs_img: (b, 256, 256, 1) => (b, 4, 4, 384)
        x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv1(x)))
        x = pad_dim(x)
        x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv2(x)))
        x = pad_dim(x)
        x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv3(x)))
        x = pad_dim(x)
        x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv4(x)))

        x = pad_dim(x)
        x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv5(x)))
        x = pad_dim(x)
        x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv6(x)))

        net = self.flatten(x)  # (b, 4*4*512)
        net = self.fc(net)

        return net


def main():
    g = Generator()
    d = Discriminator()

    # x = tf.random.uniform([128, 256, 256, 1])

    z = tf.random.normal([2, 128])

    image_fake = g(z, training=False)
    # image_fake = tf.cast((image_fake*127.5)+127.5, dtype=tf.int32)
    # image_fake = image_fake.numpy()
    # print(image_fake.shape)
    sss = d(image_fake)
    g.summary()
    d.summary()
    print(image_fake)
    print(sss)
    print(image_fake.shape)

    # print(image_fake.shape)
    # print(image_fake[0])
    # print(np.max(image_fake[0]), np.min(image_fake[0]))
    # print(np.max(image_fake[1]), np.min(image_fake[1]))
    # plt.imshow(image_fake[0], cmap='gray')
    # plt.figure()
    # plt.imshow(image_fake[1], cmap='gray')
    # plt.show()


if __name__ == '__main__':
    main()
