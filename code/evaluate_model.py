# Tensorflow Version: 1.5.0

import os
import sys
import argparse
import csv
from random import Random
import shutil

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import *
from visualization import *

np.random.seed(12345)
tf.set_random_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", help="Directory where the model weights are stored.", type=str)
parser.add_argument("--iteration", help="Model iteration to use.", type=int, default=30000)
parser.add_argument("--generate", help="Generate new images", action='store_true')
parser.add_argument("--translate", help="Translate images", action='store_true')
parser.add_argument("--interpolate", help="Interpolate between two images", action='store_true')
parser.add_argument("--reconstruct", help="Sample from the test set and reconstruct with the Encoder and Generator",
                    action='store_true')
args = parser.parse_args()

model_dir = args.model_dir+"/iteration.ckpt-"+str(args.iteration)

assert os.path.exists(model_dir+".meta"), "There exists no weight file \"iteration.ckpt-"+str(args.iteration) + "\" in {}". \
        format(args.model_dir)


def read_encodings(path):
    with open(path+"/encodings.txt") as f:
        encodings = map(int, f.read().strip().split(','))

    return encodings


def read_hyperparameters(path):
    reader = csv.reader(open(path+"/hp_file.csv", "rb"))
    dict = {}
    for row in reader:
        k, v = row
        dict[k] = v

    return dict

hp_dict = read_hyperparameters(args.model_dir)

activations = {"elu" : tf.nn.elu, "relu": tf.nn.relu, "lrelu": tf.nn.leaky_relu}

g_activation = activations[hp_dict["g_activation"]]
e_activation = activations[hp_dict["e_activation"]]
d_activation = activations[hp_dict["d_activation"]]

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

Z_DIM = int(hp_dict["num_z"])
DISC_VARS = [10]
num_disc_vars = 0
for cla in DISC_VARS:
    num_disc_vars += cla
CONT_VARS = 2
C_DIM = num_disc_vars + CONT_VARS
img_height, img_width = 28, 28
channels = 1
z_dim = int(hp_dict["num_z"])
c_dim = 12
disc_classes = [10]
num_cont_vars = 2

truncated_normal = tf.truncated_normal_initializer(stddev=0.02)
random_normal = tf.random_normal_initializer(mean=0.0, stddev=0.01)


# placeholders
X_lab = tf.placeholder(tf.float32, shape=[None, img_height, img_width, channels], name="X_lab")
Y_lab = tf.placeholder(tf.float32, shape=[None, 10], name="Y_lab")
X_unl = tf.placeholder(tf.float32, shape=[None, img_height, img_width, channels], name="Y_unl")
Y_unl = tf.placeholder(tf.float32, shape=[None, 10], name="Y_unl")
x_gen = tf.placeholder(tf.float32, shape=[None, img_height, img_width, channels], name="x_gen")
z = tf.placeholder(tf.float32, shape=[None, z_dim], name="z")
c = tf.placeholder(tf.float32, shape=[None, c_dim], name="c")
phase = tf.placeholder(tf.bool, name='phase')
num_lab = tf.placeholder(tf.int32, shape=None, name="num_lab")
adv_loss = tf.placeholder(tf.float32, shape=None, name="adv_loss")


def conv2d_bn_act(inputs, filters, kernel_size, kernel_init, activation, strides,
                  noise=False, noise_std=0.5, padding="VALID"):
    """
    Shortcut for a module of convolutional layer, batch normalization and possibly adding of Gaussian noise.
    :param inputs: input data
    :param filters: number of convolutional filters
    :param kernel_size: size of filters
    :param kernel_init: weight initialization
    :param activation: activation function (applied after batch normalization)
    :param strides: strides of the convolutional filters
    :param noise: whether to add gaussian noise to the output
    :param noise_std: stadnard deviation of added noise
    :param padding: padding in the conv layer
    :return: output data after applying the conv layer, batch norm, activation function and possibly Gaussian noise
    """
    _tmp = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            kernel_initializer=kernel_init, activation=None, strides=strides, padding=padding)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase)
    _tmp = activation(_tmp)
    if noise:
        _tmp = gaussian_noise_layer(_tmp, noise_std, phase)

    return _tmp


def deconv2d_bn_act(inputs, filters, kernel_size, kernel_init, activation, strides, padding="SAME"):
    """
        Shortcut for a module of transposed convolutional layer, batch normalization.
        :param inputs: input data
        :param filters: number of convolutional filters
        :param kernel_size: size of filters
        :param kernel_init: weight initialization
        :param activation: activation function (applied after batch normalization)
        :param strides: strides of the convolutional filters
        :param padding: padding in the conv layer
        :return: output data after applying the transposed conv layer, batch norm, and activation function
        """
    _tmp = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            kernel_initializer=kernel_init, activation=None, strides=strides, padding=padding)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase)
    _tmp = activation(_tmp)

    return _tmp


def dense_bn_act(inputs, units, activation, kernel_init, noise=False, noise_std=0.5):
    """
        Shortcut for a module of dense layer, batch normalization and possibly adding of Gaussian noise.
        :param inputs: input data
        :param units: number of units
        :param activation: activation function (applied after batch normalization)
        :param kernel_init: weight initialization
        :return: output data after applying the dense layer, batch norm, activation function and possibly Gaussian noise
        """
    _tmp = tf.layers.dense(inputs=inputs, units=units, activation=None, kernel_initializer=kernel_init)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase)
    _tmp = activation(_tmp)
    if noise:
        _tmp = gaussian_noise_layer(_tmp, noise_std, phase)

    return _tmp


# Discriminator
def discriminate(img_input, noise_input):
    with tf.variable_scope("d_net", reuse=tf.AUTO_REUSE):
        # image discriminator
        d_x_conv_0 = conv2d_bn_act(inputs=img_input, filters=64, kernel_size=3, kernel_init=he_init,
                                   activation=d_activation, strides=2, noise=True, noise_std=0.3)
        d_x_conv_1 = conv2d_bn_act(inputs=d_x_conv_0, filters=128, kernel_size=3, kernel_init=he_init,
                                   activation=d_activation, strides=2, noise=True, noise_std=0.5)
        shp = [int(s) for s in d_x_conv_1.shape[1:]]
        d_x_conv_1 = tf.reshape(d_x_conv_1, [-1, shp[0] * shp[1] * shp[2]])
        d_x_dense = dense_bn_act(inputs=d_x_conv_1, units=512, activation=d_activation, kernel_init=he_init,
                                 noise=True, noise_std=0.5)

        # noise discriminator
        noise_input = tf.reshape(noise_input, [-1, 1, 1, z_dim + c_dim])
        d_z_conv_0 = conv2d_bn_act(inputs=noise_input, filters=64, kernel_size=1, kernel_init=he_init,
                                   activation=d_activation, strides=1, noise=True, noise_std=0.3)
        d_z_conv_1 = conv2d_bn_act(inputs=d_z_conv_0, filters=128, kernel_size=1, kernel_init=he_init,
                                   activation=d_activation, strides=1, noise=True, noise_std=0.5)
        shp = [int(s) for s in d_z_conv_1.shape[1:]]
        d_z_conv_1 = tf.reshape(d_z_conv_1, [-1, shp[0] * shp[1] * shp[2]])
        d_z_dense = dense_bn_act(inputs=d_z_conv_1, units=512, activation=d_activation, kernel_init=he_init,
                                 noise=True, noise_std=0.5)

        # final discriminator
        final_input = tf.concat((d_x_dense, d_z_dense), axis=1)
        d_final_dense = dense_bn_act(inputs=final_input, units=1024, activation=d_activation, kernel_init=he_init,
                                 noise=True, noise_std=0.5)
        d_final_pred = tf.layers.dense(inputs=d_final_dense, units=1, activation=tf.nn.sigmoid,
                                       kernel_initializer=he_init)
        return d_final_pred


# Encoder
def encode(input):
    with tf.variable_scope("e_net", reuse=tf.AUTO_REUSE):
        e_conv_0 = conv2d_bn_act(inputs=input, filters=32, kernel_size=3, kernel_init=he_init,
                                 activation=e_activation, strides=1)
        e_conv_1 = conv2d_bn_act(inputs=e_conv_0, filters=64, kernel_size=3, kernel_init=he_init,
                                 activation=e_activation, strides=2)
        e_conv_2 = conv2d_bn_act(inputs=e_conv_1, filters=128, kernel_size=3, kernel_init=he_init,
                                 activation=e_activation, strides=2)
        shp = [int(s) for s in e_conv_2.shape[1:]]
        e_conv_2 = tf.reshape(e_conv_2, [-1, shp[0] * shp[1] * shp[2]])

        e_dense = dense_bn_act(inputs=e_conv_2, units=1024, activation=e_activation, kernel_init=he_init)

        # output layers
        e_dense_z = tf.layers.dense(inputs=e_dense, units=z_dim, activation=tf.nn.tanh,
                                    kernel_initializer=he_init, name="e_dense_z")
        e_dense_c_disc = []
        for idx, cla in enumerate(disc_classes):
            e_dense_c_disc.append(tf.layers.dense(inputs=e_dense, units=cla, activation=tf.nn.softmax,
                                                  kernel_initializer=he_init, name="e_dense_c_disc_" + str(idx)))
        e_dense_c_disc_concat = tf.concat(e_dense_c_disc, axis=1)
        e_dense_c_cont = tf.layers.dense(inputs=e_dense, units=num_cont_vars, activation=None,
                                         kernel_initializer=he_init, name="e_dense_c_cont")

        return e_dense_z, e_dense_c_disc_concat, e_dense_c_cont


# Generator
def generate(input):
    with tf.variable_scope("g_net", reuse=tf.AUTO_REUSE):
        g_dense_0 = dense_bn_act(inputs=input, units=3136, activation=g_activation, kernel_init=truncated_normal)
        g_dense_0 = tf.reshape(g_dense_0, [-1, 7, 7, 64])

        g_conv_0 = deconv2d_bn_act(inputs=g_dense_0, filters=128, kernel_size=4, kernel_init=truncated_normal,
                                   activation=g_activation, strides=2)

        g_conv_1 = deconv2d_bn_act(inputs=g_conv_0, filters=64, kernel_size=4, kernel_init=truncated_normal,
                                   activation=g_activation, strides=1)

        g_conv_out = tf.layers.conv2d_transpose(inputs=g_conv_1, filters=1, kernel_size=4, activation=tf.nn.sigmoid,
                                                padding='SAME', strides=2, kernel_initializer=truncated_normal)
        return g_conv_out


def generate_new_samples():
    if not os.path.exists(args.model_dir + "/samples"):
        os.makedirs(args.model_dir + "/samples")

    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    def create_image_categorical(imgs, name):
        fig = plt.figure(figsize=(0.28 * 10, 0.28 * 10))
        gs1 = gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)
        for idx1 in range(10):
            for idx2 in range(10):
                img = imgs[idx1 * 10 + idx2]
                ax = plt.subplot(gs1[idx1 * 10 + idx2])
                ax.imshow(np.reshape(img, [28, 28]), cmap='Greys_r')
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(args.model_dir + name)

    z_tmp = sample_z_fixed(128, Z_DIM)
    c_tmp = sample_c_cat(128)
    x_hat = generate(tf.concat([z, c], axis=1))

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_dir)

    gen_imgs_cat = sess.run(x_hat, feed_dict={z: z_tmp[:100], c: c_tmp[:100], phase: 0})
    create_image_categorical(gen_imgs_cat, "/samples/generated_imgs_categorical.png")

    for idx in range(2):
        z_mb = sample_z_fixed(128, z_dim)
        c_const = [_ for _ in range(2) if _ != idx]
        c_test = sample_c_cont(c_var=idx, c_const=c_const)
        gen_imgs_cont = sess.run(x_hat, feed_dict={z: z_mb, c: c_test, phase: 0})
        create_image_categorical(gen_imgs_cont, "/samples/generated_imgs_cont_"+str(idx)+".png")


def translate_samples():
    if not os.path.exists(args.model_dir + "/samples"):
        os.makedirs(args.model_dir + "/samples")

    e_dense_z, e_dense_c_disc_concat, e_dense_c_cont = encode(X_lab)
    gen_imgs = generate(tf.concat([z, c], axis=1))

    from random import Random
    py_rng = Random()

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_dir)

    # translate according to categorical values
    def sample_and_translate_categorical():
        # sample images from each class from test set
        idxs = np.arange(mnist.test.labels.shape[0])
        int_labels = [np.where(r == 1)[0][0] for r in mnist.test.labels]
        int_labels = np.asarray(int_labels)
        classes_idxs = [idxs[int_labels == y] for y in range(10)]
        sampled_idxs = [py_rng.sample(class_idxs, 1) for class_idxs in classes_idxs]
        sampled_idxs = np.asarray(sampled_idxs).flatten()

        imgs_to_be_translated = mnist.test.images[sampled_idxs]

        imgs_all = np.empty((10, 11, 28, 28))

        for idx, img in enumerate(imgs_to_be_translated):
            img = np.reshape(img, (1, 28, 28, 1))
            imgs_all[idx, 0, :, :] = np.reshape(img, (28, 28))
            _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = sess.run([e_dense_z, e_dense_c_disc_concat, e_dense_c_cont],
                                                                 feed_dict={X_lab: img, phase: 0})

            _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = np.asarray(_e_dense_z), np.asarray(_e_dense_c_disc), np.asarray(_e_dense_c_cont)
            c_tmp = np.concatenate((np.zeros((1,10)), _e_dense_c_cont), axis=1)

            for idx2 in range(10):
                c_tmp[:, :10] = 0
                c_tmp[0, idx2] = 1
                generated_img = sess.run(gen_imgs, feed_dict={z: _e_dense_z, c: c_tmp, phase: 0})
                imgs_all[idx, idx2+1, :, :] = np.reshape(generated_img, (28, 28))

        return imgs_all

    # translate according to continuous values
    def sample_and_translate_continuous_random():
        # sample images from each class from test set
        idxs = np.arange(mnist.test.labels.shape[0])
        int_labels = [np.where(r == 1)[0][0] for r in mnist.test.labels]
        int_labels = np.asarray(int_labels)
        classes_idxs = [idxs[int_labels == y] for y in range(10)]
        sampled_idxs = [py_rng.sample(class_idxs, 1) for class_idxs in classes_idxs]
        sampled_idxs = np.asarray(sampled_idxs).flatten()

        imgs_to_be_translated = mnist.test.images[sampled_idxs]

        imgs_all = np.empty((10, 7, 28, 28))

        for idx, img in enumerate(imgs_to_be_translated):
            img = np.reshape(img, (1, 28, 28, 1))
            imgs_all[idx, 0, :, :] = np.reshape(img, (28, 28))
            _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = sess.run(
                [e_dense_z, e_dense_c_disc_concat, e_dense_c_cont],
                feed_dict={X_lab: img, phase: 0})

            _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = np.asarray(_e_dense_z), np.asarray(
                _e_dense_c_disc), np.asarray(_e_dense_c_cont)
            c_tmp = np.concatenate((_e_dense_c_disc, _e_dense_c_cont), axis=1)

            for idx2 in range(2):
                _c_tmp = np.copy(c_tmp)
                _c_tmp[:, 10 + idx2] = -1
                generated_img = sess.run(gen_imgs, feed_dict={z: _e_dense_z, c: _c_tmp, phase: 0})
                imgs_all[idx, idx2 * 3 + 1, :, :] = np.reshape(generated_img, (28, 28))

                _c_tmp[:, 10 + idx2] = 0
                generated_img = sess.run(gen_imgs, feed_dict={z: _e_dense_z, c: _c_tmp, phase: 0})
                imgs_all[idx, idx2 * 3 + 2, :, :] = np.reshape(generated_img, (28, 28))

                _c_tmp[:, 10 + idx2] = 1
                generated_img = sess.run(gen_imgs, feed_dict={z: _e_dense_z, c: _c_tmp, phase: 0})
                imgs_all[idx, idx2 * 3 + 3, :, :] = np.reshape(generated_img, (28, 28))
        return imgs_all

    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    imgs_categorical = sample_and_translate_categorical()
    imgs_continuous = sample_and_translate_continuous_random()
    def create_image(imgs, name, cols):
        fig = plt.figure(figsize=(0.28 * cols, 0.28 * 10))
        gs1 = gridspec.GridSpec(10, cols, wspace=0.0, hspace=0.0)
        for idx1 in range(10):
            for idx2 in range(cols):
                img = imgs[idx1, idx2]
                ax = plt.subplot(gs1[idx1 * cols + idx2])
                ax.imshow(np.reshape(img, [28, 28]), cmap='Greys_r')
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(args.model_dir+"/samples/" + name + ".png")

    create_image(imgs_categorical, name="categorical_translations", cols=11)
    create_image(imgs_continuous, name="continuous_translations", cols=7)


def interpolate(start, finish, fix_categorical=True):
    import scipy.io as sio
    from random import Random
    py_rng = Random()
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    e_dense_z, e_dense_c_disc_concat, e_dense_c_cont = encode(X_lab)
    gen_imgs = generate(tf.concat([z, c], axis=1))

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_dir)

    labels_test = mnist.test.labels
    idxs = np.arange(labels_test.shape[0])

    int_labels = [np.where(r == 1)[0][0] for r in labels_test]
    int_labels = np.asarray(int_labels)

    classes_idxs = [idxs[int_labels == start]]
    sampled_idxs = [py_rng.sample(class_idxs, 10) for class_idxs in classes_idxs]
    sampled_idxs = np.asarray(sampled_idxs).flatten()
    imgs_start = mnist.test.images[sampled_idxs]

    classes_idxs = [idxs[int_labels == finish]]
    sampled_idxs = [py_rng.sample(class_idxs, 10) for class_idxs in classes_idxs]
    sampled_idxs = np.asarray(sampled_idxs).flatten()
    imgs_finish = mnist.test.images[sampled_idxs]


    imgs_all = np.empty((10, 11, 28, 28, 1))

    img0 = np.reshape(imgs_start, (10, 28, 28, 1))
    imgs_all[:, 0, :, :] = img0
    # encode start images
    _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = sess.run([e_dense_z, e_dense_c_disc_concat, e_dense_c_cont],
                                                            feed_dict={X_lab: img0, phase: 0})

    _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = np.asarray(_e_dense_z), np.asarray(_e_dense_c_disc), np.asarray(_e_dense_c_cont)
    z_tmp0 = _e_dense_z
    c_tmp0 = np.concatenate((_e_dense_c_disc, _e_dense_c_cont), axis=1)

    # encode end images
    img1 = np.reshape(imgs_finish, (10, 28, 28, 1))
    imgs_all[:, -1, :, :] = img1
    _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = sess.run([e_dense_z, e_dense_c_disc_concat, e_dense_c_cont],
                                                            feed_dict={X_lab: img1, phase: 0})

    _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = np.asarray(_e_dense_z), np.asarray(_e_dense_c_disc), np.asarray( _e_dense_c_cont)
    z_tmp1 = _e_dense_z
    c_tmp1 = np.concatenate((_e_dense_c_disc, _e_dense_c_cont), axis=1)

    # start interpolating between start and end images
    for idx2 in range(1, 10):
        _z = (1.0 - idx2 * 0.1) * z_tmp0 + idx2 * 0.1 * z_tmp1
        _c = (1.0 - idx2 * 0.1) * c_tmp0 + idx2 * 0.1 * c_tmp1

        if fix_categorical:
            _c[:, :10] = 0
            _c[:, start] = 1.0 - idx2 * 0.1
            _c[:, finish] = idx2 * 0.1

        generated_img = sess.run(gen_imgs, feed_dict={z: _z, c: _c, phase: 0})
        imgs_all[:, idx2, :, :] = np.reshape(generated_img, (10, 28, 28, 1))

    fig = plt.figure(figsize=(0.28 * imgs_all.shape[1], 0.28 * 10))
    gs1 = gridspec.GridSpec(10, imgs_all.shape[1], wspace=0.0, hspace=0.0)
    for idx1 in range(10):
        for idx2 in range(imgs_all.shape[1]):
            img = imgs_all[idx1, idx2]
            ax = plt.subplot(gs1[idx1*10+idx1 + idx2])
            ax.imshow(np.reshape(img, [28, 28]), cmap='Greys_r')
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.savefig(args.model_dir+"/samples/mnist_class_interpolation_" + str(start) + "_" + str(finish) + ".png")
    plt.close('all')


def reconstruct():
    import scipy.io as sio
    from random import Random
    py_rng = Random()
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    e_dense_z, e_dense_c_disc_concat, e_dense_c_cont = encode(X_lab)
    gen_imgs = generate(tf.concat([z, c], axis=1))

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_dir)

    idxs = np.arange(mnist.test.labels.shape[0])
    int_labels = [np.where(r == 1)[0][0] for r in mnist.test.labels]
    int_labels = np.asarray(int_labels)
    classes_idxs = [idxs[int_labels == y] for y in range(10)]
    sampled_idxs = [py_rng.sample(class_idxs, 1) for class_idxs in classes_idxs]
    sampled_idxs = np.asarray(sampled_idxs).flatten()

    imgs = mnist.test.images[sampled_idxs]
    imgs = np.reshape(imgs, [-1, 28, 28, 1])

    _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = sess.run([e_dense_z, e_dense_c_disc_concat, e_dense_c_cont],
                                                            feed_dict={X_lab: imgs, phase: 0})

    _e_dense_z, _e_dense_c_disc, _e_dense_c_cont = np.asarray(_e_dense_z), np.asarray(_e_dense_c_disc), np.asarray(_e_dense_c_cont)
    c_tmp = np.concatenate([_e_dense_c_disc, _e_dense_c_cont], axis=1)

    imgs_reconstructed = sess.run(gen_imgs, feed_dict={z: _e_dense_z, c: c_tmp, phase: 0})
    imgs_reconstructed = np.reshape(imgs_reconstructed, [-1, 28, 28, 1])

    imgs_all = np.concatenate([imgs, imgs_reconstructed], axis=0)

    fig = plt.figure(figsize=(0.28 * 2, 0.28 * 10))
    gs1 = gridspec.GridSpec(10, 2, wspace=0.0, hspace=0.0)
    for idx1 in range(10):
        img = imgs_all[idx1]
        ax = plt.subplot(gs1[idx1*2])
        ax.imshow(np.reshape(img, [28, 28]), cmap='Greys_r')
        ax.set_xticks([])
        ax.set_yticks([])
    for idx1 in range(10):
        img = imgs_all[idx1+10]
        ax = plt.subplot(gs1[idx1*2+1])
        ax.imshow(np.reshape(img, [28, 28]), cmap='Greys_r')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.savefig(args.model_dir + "/samples/reconstructions.png")
    plt.close('all')


if args.generate:
    generate_new_samples()
elif args.translate:
    translate_samples()
elif args.interpolate:
    start_digit = input("Enter Start Digit: ")
    end_digit = input("Enter End Digit: ")
    interpolate(start=start_digit, finish=end_digit)
elif args.reconstruct:
    reconstruct()
else:
    print("No valid option chosen. Choose either \"--generate\", \"--translate\", \"--interpolate\" or \"--reconstruct\".")
    print("Use \"--help\" for an overview of the command line arguments.")
