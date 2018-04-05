import tensorflow as tf
import numpy as np
import os
import sys
import datetime
import dateutil.tz
import argparse
from shutil import copyfile
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.layers import utils
import scipy.misc
import shutil
import time

import mnist_classifier
from utils import *
from visualization import *

# create log dir
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
log_dir = "log_dir/mnist_sem_sup/mnist_" + timestamp
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="The size of the minibatch", type=int, default=64)
parser.add_argument("--lr_d", help="Discriminator Learning Rate", type=float, default=1e-4)
parser.add_argument("--lr_g", help="Generator Learning Rate", type=float, default=5e-4)
parser.add_argument("--beta1_g", help="Generator Beta 1 (for Adam Optimizer)", type=float, default=0.5)
parser.add_argument("--beta1_d", help="Discriminator Beta 1 (for Adam Optimizer)", type=float, default=0.5)
parser.add_argument("--num_z", help="Number of noise variables", type=int, default=16)
parser.add_argument("--d_activation", help="Activation function of Discriminator", type=str, default="elu")
parser.add_argument("--g_activation", help="Activation function of Generator", type=str, default="elu")
parser.add_argument("--e_activation", help="Activation function of Encoder", type=str, default="elu")
parser.add_argument("--sup_loss", help="Weighting of the supervised loss", type=int, default=10)
parser.add_argument("--lab_sample_red", help="Reduction in probability of sampling a labeled sample", type=float, default=0.001)
parser.add_argument("--num_samples", help="Number of labeled samples per class", type=int, default=5)
parser.add_argument("--sup_loss_reduction", help="Reduction of impact of supervised loss", type=int, default=10)
parser.add_argument("--weight_recon_loss", help="Weighting of the reconstruction loss", type=int, default=1)
parser.add_argument("--max_iter", help="Maximum number of training iterations", type=int, default=50000)
parser.add_argument("--recon_loss_G", help="Number of iters recon loss applies only to G", type=int, default=0)
parser.add_argument("--gen_img_path", help="Temporary folder where imgs are generated.", type=str, default="mnist_test_imgs")
parser.add_argument("--max_adv_loss", help="Maximum impact of additional losses", type=float, default=1)
parser.add_argument("--adv_loss_increase", help="By how much additional losses are increased per iteration", type=float, default=0.001)
args = parser.parse_args()

# copy executed file and used hyperparameters to the log dir
with open(log_dir + "/hp_file.csv", "wb") as f:
    for arg in args.__dict__:
        f.write(arg + "," + str(args.__dict__[arg]) + "\n")
copyfile(sys.argv[0], log_dir + "/" + sys.argv[0])


from random import Random
# seed = 42
# py_rng = Random(seed)
py_rng = Random()

# load MNIST data set
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
mnist_imgs = mnist.train.images
mnist_labels = mnist.train.labels

# sample num_samples labeled examples for supervised training
num_samples_per_class = args.num_samples
idxs = np.arange(mnist_labels.shape[0])
int_labels = [np.where(r==1)[0][0] for r in mnist_labels]
int_labels = np.asarray(int_labels)
classes_idxs = [idxs[int_labels == y] for y in range(10)]
sampled_idxs = [py_rng.sample(class_idxs, num_samples_per_class) for class_idxs in classes_idxs]
sampled_idxs = np.asarray(sampled_idxs).flatten()

# labeled MNIST images
mnist_imgs_lab = mnist_imgs[sampled_idxs]
mnist_labels_lab = mnist_labels[sampled_idxs]

# unlabeled MNIST images
unl_idxs = [x for x in range(mnist_imgs.shape[0]) if x not in sampled_idxs]
mnist_imgs_unl = mnist_imgs[unl_idxs]
mnist_test = mnist.test.images
mnist_test = np.reshape(mnist_test, [10000, 28, 28, 1])

# hyperparameters
mb_size = args.batch_size           # batch size
img_width, img_height = 28, 28      # image size
channels = 1                        # number of channels
lr_d = args.lr_d                    # learning rate of discriminator
lr_g = args.lr_g                    # learning rate of generator and encoder
beta1_d = args.beta1_d              # beta_1 value for discriminator optimizer Adam
beta1_g = args.beta1_g              # beta_1 value for generator/encoder optimizer Adam
max_iter = args.max_iter            # maximum number of training iterations
sup_loss = args.sup_loss            # weight factor of supervised loss

z_dim = args.num_z                  # dimensionality of z (entangled part of representation)
disc_classes = [10]                 # number of discrete classes
num_disc_vars = 0                   # number of discrete variables in disentangled representation
for cla in disc_classes:
    num_disc_vars += cla
num_cont_vars = 2                   # number of continuous variables in disentangled representation
c_dim = num_disc_vars + num_cont_vars   # total number of variables in disentangled representation

# activation functions
activations = {"leaky_relu": tf.nn.leaky_relu, "relu": tf.nn.relu, "elu" : tf.nn.elu}
g_activation = activations[args.g_activation]
e_activation = activations[args.e_activation]
d_activation = activations[args.d_activation]


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


# prepare image input, i.e. concatenated labeled and unlabeled samples
images_input = tf.concat((X_lab, X_unl), axis=0)

# generate new images from disentangled representation
x_hat = generate(tf.concat([z, c], axis=1))

# encodings of real images
u_hat, c_hat_disc, c_hat_cont = encode(images_input)
z_hat = tf.concat([u_hat, c_hat_disc, c_hat_cont], axis=1)
c_hat = tf.concat([c_hat_disc, c_hat_cont], axis=1)

# encodings of generated images
u_gen, c_gen_disc, c_gen_cont = encode(x_hat)
z_gen = tf.concat([u_gen, c_gen_disc, c_gen_cont], axis=1)
c_gen = tf.concat([c_gen_disc, c_gen_cont], axis=1)

# discriminiate between real and generated images and their respective representation
d_enc = discriminate(img_input=images_input, noise_input=z_hat)
d_gen = discriminate(img_input=x_hat, noise_input=tf.concat([z, c], axis=1))


# minimize crossentropy between z and E(G(z))
# continuous cross entropy
cont_stddev_e_g_z = tf.ones_like(c_gen_cont)
eps_e_g_z = (c[:, num_disc_vars:] - c_gen_cont) / (cont_stddev_e_g_z + 1e-8)
cross_entropy_cont = tf.reduce_mean(
    -tf.reduce_sum(-0.5 * np.log(2 * np.pi) - log(cont_stddev_e_g_z) - 0.5 * tf.square(eps_e_g_z), 1))
# discrete cross entropy
cross_entropy_discr = tf.reduce_mean(-tf.reduce_sum(log(c_gen_disc) * c[:, :num_disc_vars], 1))

# loss terms
# reconstruction loss
x_recon = generate(input=z_hat)
reconstruction_loss = args.weight_recon_loss*tf.reduce_mean(tf.square(x_recon - images_input))
# adversarial loss
G_E_loss = -tf.reduce_mean(log(d_gen) + log(1 - d_enc))
D_loss = -tf.reduce_mean(log(d_enc) + log(1 - d_gen))
# supervised loss
G_loss_sup = tf.reduce_mean(-tf.reduce_sum(log(c_hat_disc[:num_lab, :disc_classes[0]]) * Y_lab, 1))
supervised_loss = args.sup_loss * G_loss_sup
# mutual information maximization loss
G_E_loss_cross_ent = cross_entropy_cont + cross_entropy_discr
# total loss for G and E
final_loss = reconstruction_loss + adv_loss * G_E_loss_cross_ent + adv_loss*G_E_loss + supervised_loss

# training operations
all_vars = tf.trainable_variables()
theta_G = [var for var in all_vars if var.name.startswith('g_') or var.name.startswith('e_')]
theta_D = [var for var in all_vars if var.name.startswith('d_')]
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step, important for updating
    # the batch normalization parameters
    with tf.control_dependencies(update_ops):
        boundaries = [10000, 20000, 30000, 40000]
        global_step_d = tf.Variable(0, trainable=False)
        global_step_g = tf.Variable(0, trainable=False)
        values = [lr_d, lr_d / 2.0, lr_d / 4.0, lr_d / 8.0, lr_d / 10.0]
        lr_dis = tf.train.piecewise_constant(global_step_d, boundaries, values)
        D_solver = (tf.train.AdamOptimizer(learning_rate=lr_dis, beta1=beta1_d)
                    .minimize(D_loss, var_list=theta_D, global_step=global_step_d))

        values = [lr_g, lr_g / 2.0, lr_g / 4.0, lr_g / 8.0, lr_g / 10.0]
        lr_gen = tf.train.piecewise_constant(global_step_g, boundaries, values)
        G_E_solver = (tf.train.AdamOptimizer(learning_rate=lr_gen, beta1=beta1_g)
                    .minimize(final_loss, var_list=theta_G, global_step=global_step_g))

# scalar logging for Tensorboard
supervised_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(c_hat_disc[:num_lab, :disc_classes[0]],1), tf.argmax(Y_lab,1)), tf.float32))
g_cross_entropy_loss = cross_entropy_discr + cross_entropy_cont
tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_E_loss", G_E_loss)
tf.summary.scalar("G_loss", -tf.reduce_mean(log(d_gen)))
tf.summary.scalar("E_loss", -tf.reduce_mean(log(1 - d_enc)))
tf.summary.scalar("D_data_acc", tf.reduce_mean(d_enc))
tf.summary.scalar("D_gen_acc", tf.reduce_mean(1 - d_gen))
tf.summary.scalar("E_sup_acc", supervised_accuracy)
tf.summary.scalar("cross_entropy_cont", cross_entropy_cont)
tf.summary.scalar("cross_entropy_discr", cross_entropy_discr)
summary_op = tf.summary.merge_all()

# image logging for Tensorboard
summary_categorical_c = summary_cat(x_hat)
summary_continuous_c = summary_cont(x_hat)
summary_reconstruction = summary_reconstruction(X_unl, x_hat)


# generate num_samples images of each class to test generator performance
def generate_test_imgs(num_samples=500, img_path = args.gen_img_path):
    if os.path.exists(img_path):
        shutil.rmtree(img_path)
    os.makedirs(img_path)

    for idx in range(10):
        os.makedirs(img_path + "/" + str(idx))
        _z = sample_z(num_samples, z_dim)
        _c = np.zeros((num_samples, c_dim))
        _c[:, idx] = 1

        for n in range(num_cont_vars):
            cont = np.random.uniform(-1, 1, size=(num_samples))
            _c[:, 10+n] = cont

        for _ in range(num_samples/500):
            generated_imgs = (sess.run(x_hat, feed_dict={z: _z[_*500:_*500+500], c: _c[_*500:_*500+500], phase: 0}))
            generated_imgs = np.reshape(generated_imgs, (-1, 28, 28))
            for img_id, img in enumerate(generated_imgs):
                scipy.misc.imsave(
                    img_path + '/' + str(idx) + '/class' + str(idx) + '_' + str(_) + '_' + str(img_id) + '.jpg', img)


# calculate accuracy of encoder on the test set
def calculate_test_set_accuracy():
    _num_lab = 0
    X_labeled, Y_labeled = sample_mnist(0, labeled=True)

    z_representations = []
    for idx in range(10):
        X_unlabeled, Y_unlabeled = mnist.test.next_batch(1000, shuffle=False)
        X_unlabeled = np.reshape(X_unlabeled, (-1, 28, 28, 1))

        z, e_disc, e_cont = sess.run([u_hat, c_hat_disc, c_hat_cont],
                                     feed_dict={X_unl: X_unlabeled, Y_unl: Y_unlabeled,
                                                X_lab: X_labeled, Y_lab: Y_labeled,
                                                num_lab: _num_lab, phase: 0})
        z, e_disc, e_cont = np.asarray(z), np.asarray(e_disc), np.asarray(e_cont)
        repr = np.concatenate((z, e_disc, e_cont), axis=1)
        z_representations.extend(repr)

    z_representations = np.asarray(z_representations)
    z_representations = np.reshape(z_representations, (10000, z_dim + c_dim))

    # predictions = []
    # for idx in range(z_representations.shape[0]):
    #     predictions.append(np.argmax(z_representations[idx, z_dim:z_dim + num_disc_vars]))
    # predictions = np.asarray(predictions)
    predictions = np.argmax(z_representations[:, z_dim:z_dim + num_disc_vars], axis=1)

    mnist_labels = np.argmax(mnist.test.labels, axis=1)

    acc = np.mean(np.equal(predictions, mnist_labels))

    return acc


# sample batch from MNIST data set
def sample_mnist(mb_size, labeled=False):
    if labeled:
        rand = np.random.randint(mnist_imgs_lab.shape[0], size=mb_size)
        batch = mnist_imgs_lab[rand]
        batch = np.reshape(batch, (-1, 28, 28, 1))
        labels = mnist_labels_lab[rand]
        return batch, labels
    else:
        rand = np.random.randint(mnist_imgs_unl.shape[0], size=mb_size)
        batch = mnist_imgs_unl[rand]
        batch = np.reshape(batch, (-1, 28, 28, 1))
        return batch, np.zeros((batch.shape[0], 10))


sess = tf.Session()
saver = tf.train.Saver(max_to_keep=2)
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
print("Initialized new session")

# start with prob 1 of sampling labeled examples, is reduced after each iteration
labeled_sampling_prob = 1
# initial accuracy of the generator
generator_accuracy = 0.0
# initial weight factor of adversarial loss, is increased after each iteration
_adv_loss = 0.0
global_step = 0
for it in range(global_step + 1, global_step + max_iter + 1):
    # check if the probability of sampling a labeled example needs to be reduced (initial value is 1)
    if labeled_sampling_prob > num_samples_per_class*10/55000.0:
        labeled_sampling_prob -= args.lab_sample_red
        if labeled_sampling_prob < num_samples_per_class*10/55000.0:
            labeled_sampling_prob = num_samples_per_class*10/55000.0

    # check if the impact of the adversarial loss needs to be increased (initial value is 0)
    if _adv_loss < args.max_adv_loss:
        _adv_loss += args.adv_loss_increase

    # decide how many labeled samples to use for this training iteration
    lab_samples = np.random.binomial(mb_size, labeled_sampling_prob)
    # sample labeled and unlabeled images
    X_labeled, Y_labeled = sample_mnist(lab_samples, labeled=True)
    X_unlabeled, Y_unlabeled = sample_mnist(mb_size-lab_samples, labeled=False)
    # sample random noise for generator
    z_mb = sample_z(mb_size, z_dim)
    c_mb = sample_c(mb_size, test=False)

    # perform training step
    _, _, summary = sess.run(
        [D_solver, G_E_solver, summary_op],
        feed_dict={X_lab: X_labeled, Y_lab: Y_labeled, X_unl: X_unlabeled, Y_unl: Y_unlabeled,
                   z: z_mb, c: c_mb, phase: 1, adv_loss: _adv_loss, num_lab: lab_samples})

    # check and log generator and encoder accuracy
    if it % 1000 == 0:
        lab_samples = mb_size

        X_labeled, Y_labeled = sample_mnist(mb_size, labeled=True)
        X_unlabeled, Y_unlabeled = sample_mnist(0, labeled=False)

        _acc, _sup_loss, _cross_ent_loss, _adv_loss_, _rec_loss = sess.run(
            [supervised_accuracy, G_loss_sup, g_cross_entropy_loss, adv_loss, reconstruction_loss],
            feed_dict={X_lab: X_labeled, Y_lab: Y_labeled, X_unl: X_unlabeled, Y_unl: Y_unlabeled,
                       z: z_mb, c: c_mb, phase: 0, adv_loss: _adv_loss, num_lab: lab_samples})

        # generate test images
        generate_test_imgs(num_samples=500)
        # calculate accuracy of pretrained MNIST classifier on the generated images
        generator_accuracy_current = mnist_classifier.predict(num_samples=500, img_path=args.gen_img_path)
        # calculate encoder accuracy on test set
        encoder_accuracy = calculate_test_set_accuracy()

        # if generator accuracy improved store model weights
        if generator_accuracy_current > generator_accuracy:
            generator_accuracy = generator_accuracy_current
            fn = saver.save(sess, "{}/iteration.ckpt".format(log_dir), global_step=it)
            print("Saved model at iteration: {} with generator accuracy of {}".format(it, generator_accuracy))

        # log current generator and encoder accuracies and other information
        with open(log_dir + "/info.txt", "ab") as logfile:
            logfile.write(
                "time: {}, iter: {}, supervised accuracy: {}, G sup loss: {}, G cross entropy loss: {}, adversarial loss: {}, "
                "reconstruction loss: {}\n".
                format(time.ctime(), it, _acc, _sup_loss, _cross_ent_loss, _adv_loss_, _rec_loss))
            logfile.write("\t generator accuracy: {} \n".format(generator_accuracy_current))
            logfile.write("\t encoder accuracy: {} \n".format(encoder_accuracy))

    # run summary ops for logging in Tensorboard
    if it % 2500 == 0:
        lab_samples = 0
        X_labeled, Y_labeled = sample_mnist(0, labeled=True)
        X_unlabeled, Y_unlabeled = sample_mnist(mb_size, labeled=False)

        # vary categorical variables in c
        summary_disc = []
        for idx in range(len(disc_classes)):
            z_mb = sample_z_fixed(128, z_dim)
            c_test = sample_c(128, test=True, disc_var=idx)
            z_tmp, _, summary_d = sess.run([z_hat, x_hat, summary_categorical_c[idx]],
                                           feed_dict={X_lab: X_labeled, Y_lab: Y_labeled,
                                                      X_unl: X_unlabeled, Y_unl: Y_unlabeled, z: z_mb, c: c_test,
                                                      phase: 0, adv_loss: _adv_loss, num_lab: lab_samples})
            summary_disc.append(summary_d)

        # visualize image reconstructions
        z_tmp = np.asarray(z_tmp)
        _, _, summary_reconstruct = sess.run([z_hat, x_hat, summary_reconstruction],
                                             feed_dict={X_lab: X_labeled, Y_lab: Y_labeled,
                                                        X_unl: X_unlabeled, Y_unl: Y_unlabeled,
                                                        z: z_tmp[:, :z_dim], c: z_tmp[:, z_dim:],
                                                        adv_loss: _adv_loss, phase: 0, num_lab: lab_samples})

        # vary continuous variables in c
        summary_cont = []
        for idx in range(num_cont_vars):
            z_mb = sample_z_fixed(128, z_dim)
            c_const = [_ for _ in range(num_cont_vars) if _ != idx]
            c_test = sample_c_cont(c_var=idx, c_const=c_const)
            _, _, summary_c = sess.run([z_hat, x_hat, summary_continuous_c[idx]],
                                       feed_dict={X_lab: X_labeled, Y_lab: Y_labeled,
                                                  X_unl: X_unlabeled, Y_unl: Y_unlabeled, z: z_mb, c: c_test,
                                                  phase: 0, adv_loss: _adv_loss, num_lab: lab_samples})
            summary_cont.append(summary_c)

        # store summaries to Tensorboard
        summary_writer.add_summary(summary, it)
        for summ in summary_disc:
            summary_writer.add_summary(summ, it)
        for summ in summary_cont:
            summary_writer.add_summary(summ, it)
        summary_writer.add_summary(summary_reconstruct, it)
        summary_writer.flush()
