import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import glob, datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mat73

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


def loaddata(filepath):
    geo=mat73.loadmat(filepath+"/geo.mat")
    geo=geo["geo"]

    rho = loadmat(filepath+"/rho.mat")
    rho = rho["rho"]
    young = loadmat(filepath+"/Young.mat")
    young = young["Young"]/150
    diffusion = loadmat(filepath+"/diffusion_average.mat")
    diffusion = diffusion["diffusion_average"]

    # Random shuffling images and labels
    np.random.seed(9487)
    indice = np.array(range(len(geo)))
    np.random.shuffle(indice)
    geo = geo[indice]
    geo = tf.cast(tf.expand_dims(geo, -1), tf.float64)
    labels = np.concatenate((rho, young, diffusion), -1)
    labels = tf.cast(labels[indice], tf.float64)
    return geo, labels


def new_tri_plot():
    # lims_young = [0, 12]
    # lims_poi = [-0.5, 0.5]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[12, 4], constrained_layout=True)

    # ax1.plot(lims_young, lims_young, 'gray')
    # ax1.set_xlim(lims_young)
    # ax1.set_ylim(lims_young)
    ax1.set_aspect(1)
    ax1.set_xlabel("True values")
    ax1.set_ylabel("Predictions")
    ax1.set_title("rho")

    # ax2.plot(lims_poi, lims_poi, 'gray')
    # ax2.set_xlim(lims_poi)
    # ax2.set_ylim(lims_poi)
    ax2.set_aspect(1)
    ax2.set_xlabel("True values")
    ax2.set_ylabel("Predictions")
    ax2.set_title("Young's modulus")

    ax3.set_aspect(1)
    ax3.set_xlabel("True values")
    ax3.set_ylabel("Predictions")
    ax3.set_title("diffusion")

    return ax1, ax2, ax3


def tri_plot(y, pred, ax1, ax2, ax3):

    x1 = y[:, 0]
    x2 = y[:, 1]
    x3 = y[:, 2]
    y1 = pred[:, 0]
    y2 = pred[:, 1]
    y3 = pred[:, 2]

    ax1.scatter(x1, y1, c='C0', alpha=0.2)
    ax2.scatter(x2, y2, c='C1', alpha=0.2)
    ax3.scatter(x3, y3, c='C2', alpha=0.2)


def pad_dim(x, n=1):
    x = tf.concat((x[:, :, :, -n:, :], x, x[:, :, :, :n, :]), axis=-2)
    x = tf.concat((x[:, :, -n:, :, :], x, x[:, :, :n, :, :]), axis=-3)
    x = tf.concat((x[:, -n:, :, :, :], x, x[:, :n, :, :, :]), axis=-4)
    return x


class Solver(keras.Model):

    def __init__(self):
        super(Solver, self).__init__()

        # unit 1, [64,64,64,1] => [32,32,32,16]
        self.conv1a = layers.Conv3D(16, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv1b = layers.Conv3D(16, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max1 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 2, => [16,16,16,32]
        self.conv2a = layers.Conv3D(32, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv2b = layers.Conv3D(32, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max2 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 3, => [8,8,8,64]
        self.conv3a = layers.Conv3D(64, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv3b = layers.Conv3D(64, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max3 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 4, => [4,4,4,128]
        self.conv4a = layers.Conv3D(128, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv4b = layers.Conv3D(128, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max4 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 5, => [2,2,2,256]
        self.conv5a = layers.Conv3D(256, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv5b = layers.Conv3D(256, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max5 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 6, => [1,1,1,512]
        self.conv6a = layers.Conv3D(512, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv6b = layers.Conv3D(512, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max6 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 7, => [2]
        self.fc1 = layers.Dense(256, activation=tf.nn.relu)
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)
        self.fc3 = layers.Dense(3, activation=None)

    def call(self, x):
        # inputs_noise: (b, 64), inputs_condition: (b, 3)
        x = pad_dim(x)
        x = self.conv1a(x)
        x = pad_dim(x)
        x = self.conv1b(x)
        x = self.max1(x)

        x = pad_dim(x)
        x = self.conv2a(x)
        x = pad_dim(x)
        x = self.conv2b(x)
        x = self.max2(x)

        x = pad_dim(x)
        x = self.conv3a(x)
        x = pad_dim(x)
        x = self.conv3b(x)
        x = self.max3(x)

        x = pad_dim(x)
        x = self.conv4a(x)
        x = pad_dim(x)
        x = self.conv4b(x)
        x = self.max4(x)

        x = pad_dim(x)
        x = self.conv5a(x)
        x = pad_dim(x)
        x = self.conv5b(x)
        x = self.max5(x)

        x = pad_dim(x)
        x = self.conv6a(x)
        x = pad_dim(x)
        x = self.conv6b(x)
        x = self.max6(x)

        x = tf.keras.layers.Flatten()(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    # x = tf.expand_dims(x, -1)
    y = tf.cast(y, dtype=tf.float32)
    return x, y


def main():

    # tf.random.set_seed(2345)

    filepath = r'/home/xiaoyang/PycharmProjects/pythonProject30_3d_foam/cgan20220421/dataset'
    dataset_matrixes, dataset_labels = loaddata(filepath)
    data_length = dataset_labels.shape[0]


    x_test = dataset_matrixes[int(0.8 * len(dataset_matrixes)):-1]
    y_test = dataset_labels[int(0.8 * len(dataset_labels)):-1]
    x = dataset_matrixes[0:int(0.8 * len(dataset_matrixes))]
    y = dataset_labels[0:int(0.8 * len(dataset_labels))]

    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(10000).map(preprocess).batch(32)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess).batch(32)

    solver = Solver()
    solver.build(input_shape=[None, 64, 64, 64, 1])


    optimizer = optimizers.Adam(learning_rate=1e-4, beta_1=0.5)  ## lr = 2e-4, beta_1 = 0.9

    # plot test
    ax1, ax2, ax3 = new_tri_plot()
    total_sum = 0
    total_error = 0
    for x, y in test_db:
        pred = solver(x)
        loss_test = tf.losses.mean_squared_error(y, pred)
        loss_test = tf.reduce_mean(loss_test)

        total_sum += 1
        total_error += loss_test

        tri_plot(y, pred, ax1, ax2, ax3)

    mse = total_error / total_sum
    print(0, "mse: ", mse.numpy())
    plt.savefig('results/test/%d_test.png' % 0)
    plt.close()
    with summary_writer.as_default():
        tf.summary.scalar('loss:', float(mse.numpy()), step=0)
        tf.summary.scalar('mse', float(mse.numpy()), step=0)


    for epoch in range(200):

        # plot train
        ax1, ax2, ax3 = new_tri_plot()
        epoch += 1
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                logits = solver(x)
                loss = tf.losses.mean_squared_error(y, logits)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, solver.trainable_variables)
            optimizer.apply_gradients(zip(grads, solver.trainable_variables))



            if step < 300:
                tri_plot(y, logits, ax1, ax2, ax3)

        plt.savefig('results/train/%d_train.png' % epoch)
        plt.close()
        print(epoch, "loss: ", loss.numpy())

        # plot test
        ax1, ax2, ax3 = new_tri_plot()
        total_sum = 0
        total_error = 0
        for x, y in test_db:

            pred = solver(x)
            loss_test = tf.losses.mean_squared_error(y, pred)
            loss_test = tf.reduce_mean(loss_test)

            total_sum += 1
            total_error += loss_test

            tri_plot(y, pred, ax1, ax2, ax3)

        mse = total_error / total_sum
        print(epoch, "mse: ", mse.numpy())
        plt.savefig('results/test/%d_test.png' % epoch)
        plt.close()

        with summary_writer.as_default():
            tf.summary.scalar('loss:', float(loss.numpy()), step=epoch)
            tf.summary.scalar('mse', float(mse.numpy()), step=epoch)

        solver.save_weights('ckpt/solver_%d.ckpt' % epoch)


if __name__ == "__main__":
    main()
