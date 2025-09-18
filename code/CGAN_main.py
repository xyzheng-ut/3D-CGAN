import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from PIL import Image
import glob, datetime
from pure_gan_kernel4 import Generator, Discriminator
from regression_with_ckpt import Solver
from load_data import load_data
import matplotlib.pyplot as plt
import mat73
import math

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)



def new_tri_plot():
    lims_rho = [0.2, 0.8]
    lims_young = [0, 1]
    lims_diffusion = [0.1, 0.4]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[12, 4], constrained_layout=True)

    ax1.plot(lims_rho, lims_rho, 'gray')
    ax1.set_xlim(lims_rho)
    ax1.set_ylim(lims_rho)
    ax1.set_aspect(1)
    ax1.set_xlabel("Input")
    ax1.set_ylabel("Output")
    ax1.set_title("rho")

    ax2.plot(lims_young, lims_young, 'gray')
    ax2.set_xlim(lims_young)
    ax2.set_ylim(lims_young)
    ax2.set_aspect(1)
    ax2.set_xlabel("Input")
    ax2.set_ylabel("Output")
    ax2.set_title("Young's modulus")

    ax3.plot(lims_diffusion, lims_diffusion, 'gray')
    ax3.set_xlim(lims_diffusion)
    ax3.set_ylim(lims_diffusion)
    ax3.set_aspect(1)
    ax3.set_xlabel("Input")
    ax3.set_ylabel("Output")
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


def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]

    # Label Smoothing, replace the label with a random number between 0.7 and 1.2
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                          (logits=logits, labels=tf.ones_like(logits)-0.3 + np.random.uniform(size=logits.shape) * 0.5))
    # loss = tf.keras.losses.categorical_crossentropy(y_pred=logits,
    #                                                 y_true=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                          (logits=logits, labels=tf.zeros_like(logits)+np.random.uniform(size=logits.shape) * 0.3))
    # loss = tf.keras.losses.categorical_crossentropy(y_pred=logits,
    #                                                 y_true=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, real_seeds, fake_seeds):

    alpha = tf.random.uniform(shape=real_seeds.get_shape(), minval=0., maxval=1.)
    differences = fake_seeds - real_seeds  # This is different from MAGAN
    interpolates = real_seeds + (alpha * differences)
    with tf.GradientTape() as tape:
        tape.watch([interpolates])
        d_interplote_logits = discriminator(interpolates, training=True)
    grads = tape.gradient(d_interplote_logits, interpolates)

    # grads:[b, 64, 2] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp


def d_loss_fn(generator, discriminator, batch_z, batch_x, condition, is_training):
    # 1. treat real image as real
    # 2. treat generated image as fake
    fake_image = generator(batch_z, condition, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_zeros(d_real_logits)
    d_loss_fake = celoss_ones(d_fake_logits)
    gp = gradient_penalty(discriminator, batch_x, fake_image)

    loss = d_loss_real + d_loss_fake + 10 * gp

    return loss, gp


def g_loss_fn(generator, discriminator, solver, batch_z, condition, is_training):
    fake_image = generator(batch_z, condition, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_zeros(d_fake_logits)

    reference = solver(fake_image)
    mse = tf.reduce_mean(tf.losses.mean_squared_error(condition, reference))

    return loss+0.1*mse, mse


def preprocess(x):
    x = tf.cast(x, dtype=tf.float32)
    return x


def get_data_space(batch_size):
    rho = np.random.uniform(low=0.2, high=0.8, size=[batch_size, 1])
    young = (250.9/140) * rho**2.845
    diffusion = rho * 0.5
    return np.concatenate((rho, young, diffusion), axis=-1)


def main():
    tf.random.set_seed(222)
    np.random.seed(222)

    assert tf.__version__.startswith('2.')

    # hyper parameters
    z_dim = 128
    epochs = 200
    batch_size = 32
    learning_rate = 0.0001
    is_training = True

    generator = Generator()
    # generator.load_weights(r'/home/xiaoyang/Desktop/pycharm_project/project10_matrix64/ckpt/20210527-181242_smoothing lable_3rd/generator_133.ckpt')

    discriminator = Discriminator()
    # discriminator.load_weights(r'/home/xiaoyang/Desktop/pycharm_project/project10_matrix64/ckpt/20210527-181242_smoothing lable_3rd/discriminator_133.ckpt')

    solver = Solver()
    solver.load_weights(r"ckpt/solver_199.ckpt")
    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):

        mat = mat73.loadmat(r'/home/xiaoyang/PycharmProjects/pythonProject30_3d_foam/gan20220415/mat/geo.mat')
        data_npy = mat["geo"]
        data_npy = np.expand_dims(data_npy,-1)
        print(data_npy.shape)

        dataset = tf.data.Dataset.from_tensor_slices(data_npy)
        dataset = dataset.map(preprocess).shuffle(1000)
        dataset = dataset.batch(batch_size)


        # train G for 2 times
        for step, real_images in enumerate(dataset):
            batch = len(real_images)
            for _ in range(1):
                noise = tf.random.normal([batch, z_dim])  # (b, 128)
                condition = get_data_space(batch)

                # E and nu => E and G => [-1~1]
                with tf.GradientTape() as tape:
                    d_loss, gp = d_loss_fn(generator, discriminator, batch_z=noise, batch_x=real_images,
                                           condition=condition, is_training=is_training)
                if d_loss>0.7:
                    grads = tape.gradient(d_loss, discriminator.trainable_variables)
                    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            for _ in range(1):
                # E and nu => E and G => [-1~1]
                noise = tf.random.normal([batch, z_dim])
                condition = get_data_space(batch)
                with tf.GradientTape() as tape:
                    g_loss, mse = g_loss_fn(generator, discriminator, solver, batch_z=noise,
                                            condition=condition, is_training=is_training)
                grads = tape.gradient(g_loss, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

            for _ in range(epoch//7+1):
                condition = get_data_space(batch)
                noise = tf.random.normal([batch, z_dim])
                with tf.GradientTape() as tape:
                    fake_image = generator(noise, condition, is_training)
                    reference = solver(fake_image)
                    mse = tf.reduce_mean(tf.losses.mean_squared_error(condition, reference))
                grads = tape.gradient(mse, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))


        # plot test

        ax1, ax2, ax3 = new_tri_plot()
        total_sum = 0
        total_error = 0
        for i in range(10):
            noise_test = tf.random.normal([16, z_dim])
            condition = get_data_space(16)

            fake_image = generator(noise_test, condition, training=False)
            pred = solver(fake_image)
            loss_test = tf.reduce_mean(tf.losses.mean_squared_error(condition, pred))

            total_sum += 1
            total_error += loss_test

            tri_plot(condition, pred, ax1, ax2, ax3)

        # save_generated_voxels
        np.save("fake_image/%d.npy" % epoch, np.round(fake_image).astype("bool"))
        mse = total_error / total_sum

        plt.savefig('results/test/%d_test.png' % epoch)
        plt.close()

        print(epoch, 'd-loss: ', float(d_loss), 'gp:', float(gp), 'g-loss', float(g_loss), "mse: ", float(mse.numpy()))
        # vusulize it on tensorboard: http://localhost:6006/

        with summary_writer.as_default():

            tf.summary.scalar('d-loss: ', float(d_loss), step=epoch)
            tf.summary.scalar('gp: ', float(gp), step=epoch)
            tf.summary.scalar('g-loss: ', float(g_loss), step=epoch)
            tf.summary.scalar('mse: ', float(mse.numpy()), step=epoch)

        # save weights
        generator.save_weights('ckpt/generator_%d.ckpt' % (epoch))
        discriminator.save_weights('ckpt/discriminator_%d.ckpt' % epoch)


if __name__ == '__main__':
    main()
