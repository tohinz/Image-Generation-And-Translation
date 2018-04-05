import numpy as np
import tensorflow as tf
from tensorflow.python.layers import utils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def log(x):
    return tf.log(x + 1e-8)


# weight initialization functions
def he_init(size, dtype=tf.float32, partition_info=None):
    in_dim = size[0]
    he_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=he_stddev)

truncated_normal = tf.truncated_normal_initializer(stddev=0.02)


def gaussian_noise_layer(input_layer, std, training):
    def add_noise():
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    return utils.smart_cond(training, add_noise, lambda: input_layer)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def sample_z_fixed(m, n):
    z_out = np.zeros((m, n))
    for idx in range(10):
        z_out[idx * 10:idx * 10 + 10, :] = np.random.uniform(-1., 1., size=[1, n])
    return z_out


def sample_c(m, test=False, disc_var=0, num_cont_vars=2, disc_classes=[10], num_disc_vars=10):
    """sample a random c value
    if test is True, samples a value for each discrete variable and combines each with the chosen continuous
    variable c; all other continuous c variables are sampled once and then kept fixed"""
    if test:
        # cont = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        cont = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        cont = []
        cont_matr = []
        for idx in range(num_cont_vars):
            cont.append(np.random.uniform(-1, 1, size=[1, 10]))
            cont_matr.append(np.zeros(shape=(m)))
        #
        for idx in range(num_cont_vars):
            for idx2 in range(10):
                cont_matr[idx][10 * idx2: 10 * idx2 + 10] = np.broadcast_to(cont[idx][0, idx2], (10))

        cs_cont = np.zeros(shape=(128, num_cont_vars))

        i = 0
        for idx in range(num_cont_vars):
            cs_cont[:, idx] = cont_matr[idx]

        c = np.eye(10, 10)
        for idx in range(1, 10):
            c_tmp = np.eye(10, 10)
            c = np.concatenate((c, c_tmp), axis=0)

        counter = 0
        cs_disc = np.zeros(shape=(100, num_disc_vars))
        for idx, cla in enumerate(disc_classes):
            if idx == disc_var:
                tmp = np.eye(cla, cla)
                tmp_ = np.eye(cla, cla)
                for idx2 in range(100 / cla - 1):
                    tmp = np.concatenate((tmp, tmp_), axis=0)
                cs_disc[:, counter:counter + cla] = tmp
                counter += cla
            else:
                rand = np.random.randint(0, cla)
                tmp = np.zeros(shape=(100, cla))
                tmp[:, rand] = 1
                cs_disc[:, counter:counter + cla] = tmp
                counter += cla

        zeros = np.zeros((28, num_disc_vars))
        cs_disc = np.concatenate((cs_disc, zeros), axis=0)

        c = np.concatenate((cs_disc, cs_cont), axis=1)

        return c
    else:
        c = np.random.multinomial(1, disc_classes[0] * [1.0 / disc_classes[0]], size=m)
        for cla in disc_classes[1:]:
            c = np.concatenate((c, np.random.multinomial(1, cla * [1.0 / cla], size=m)), axis=1)
        for n in range(num_cont_vars):
            cont = np.random.uniform(-1, 1, size=(m, 1))
            c = np.concatenate((c, cont), axis=1)
        return c


# def sample_c(m, num_cont_vars=2, disc_classes=[10]):
#     """
#     Sample a random c vector, i.e. categorical and continuous variables.
#     If test is True, samples a value for each discrete variable and combines each with the chosen continuous
#     variable c; all other continuous c variables are sampled once and then kept fixed
#     """
#     c = np.random.multinomial(1, disc_classes[0] * [1.0 / disc_classes[0]], size=m)
#     for cla in disc_classes[1:]:
#         c = np.concatenate((c, np.random.multinomial(1, cla * [1.0 / cla], size=m)), axis=1)
#     for n in range(num_cont_vars):
#         cont = np.random.uniform(-1, 1, size=(m, 1))
#         c = np.concatenate((c, cont), axis=1)
#     return c


# tt = sample_c(128, test=False)
# print(tt[20:40])
# exit()

# def sample_c_cont(c_var=0, c_const=[1]):
#     z = []
#     for idx in range(len(c_const)):
#         z.append(np.random.uniform(-1, 1))
#     # cont = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
#     cont = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0]
#
#     c_cont = np.zeros((10, num_cont_vars))
#     i = 0
#     for idx in range(num_cont_vars):
#         if idx == c_var:
#             c_cont[:, c_var] = cont
#         else:
#             c_cont[:, idx] = z[i]
#             i += 1
#
#     c_out = np.zeros((128, c_dim))
#     for idx in range(10):
#         c_out[10 * idx:10 * idx + 10, idx] = 1
#         c_out[10 * idx:10 * idx + 10, num_disc_vars:] = c_cont
#
#     return c_out

def sample_c_cat(m, disc_var=0, num_cont_vars=2, num_disc_vars=10, disc_classes=[10]):
    """
    Samples categorical values for visualization purposes
    """
    cont = []
    cont_matr = []
    for idx in range(num_cont_vars):
        cont.append(np.random.uniform(-1, 1, size=[1, 10]))
        cont_matr.append(np.zeros(shape=(m)))

    for idx in range(num_cont_vars):
        for idx2 in range(10):
            cont_matr[idx][10 * idx2: 10 * idx2 + 10] = np.broadcast_to(cont[idx][0, idx2], (10))

    cs_cont = np.zeros(shape=(128, num_cont_vars))

    for idx in range(num_cont_vars):
        cs_cont[:, idx] = cont_matr[idx]

    c = np.eye(10, 10)
    for idx in range(1, 10):
        c_tmp = np.eye(10, 10)
        c = np.concatenate((c, c_tmp), axis=0)

    counter = 0
    cs_disc = np.zeros(shape=(100, num_disc_vars))
    for idx, cla in enumerate(disc_classes):
        if idx == disc_var:
            tmp = np.eye(cla, cla)
            tmp_ = np.eye(cla, cla)
            for idx2 in range(100 / cla - 1):
                tmp = np.concatenate((tmp, tmp_), axis=0)
            cs_disc[:, counter:counter + cla] = tmp
            counter += cla
        else:
            rand = np.random.randint(0, cla)
            tmp = np.zeros(shape=(100, cla))
            tmp[:, rand] = 1
            cs_disc[:, counter:counter + cla] = tmp
            counter += cla

    zeros = np.zeros((28, num_disc_vars))
    cs_disc = np.concatenate((cs_disc, zeros), axis=0)

    c = np.concatenate((cs_disc, cs_cont), axis=1)

    return c


def sample_c_cont(num_cont_vars=2, num_disc_vars=10, c_dim=12, c_var=0, c_const=[1]):
    """
    Samples continuous values for visualization purposes
    """
    z = []
    for idx in range(len(c_const)):
        z.append(np.random.uniform(-1, 1))
    cont = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0]

    c_cont = np.zeros((10, num_cont_vars))
    i = 0
    for idx in range(num_cont_vars):
        if idx == c_var:
            c_cont[:, c_var] = cont
        else:
            c_cont[:, idx] = z[i]
            i += 1

    c_out = np.zeros((128, c_dim))
    for idx in range(10):
        c_out[10 * idx:10 * idx + 10, idx] = 1
        c_out[10 * idx:10 * idx + 10, num_disc_vars:] = c_cont

    return c_out


def sample_disc_test_set(encodings, it):
    def _update(d_rep, d_act, d_idx, idx):
        d_class = get_max_idx(max(d_rep), d_rep)
        _d_act = max(d_rep)
        if _d_act > min(d_act[d_class]):
            argmin = np.argmin(d_act[d_class])
            d_act[d_class, argmin] = _d_act
            d_idx[d_class, argmin] = idx
        return d_act, d_idx

    disc1_act, disc1_idx = np.zeros((10, 10)), np.zeros((10, 10))

    for idx, rep in enumerate(encodings):
        rep = rep[z_dim:]
        disc1_act, disc1_idx = _update(rep[:10], disc1_act, disc1_idx, idx)

    def create_image(d_idx, name):
        f, axarr = plt.subplots(10, 10)
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
        for idx1 in range(10):
            for idx2 in range(10):
                id = int(d_idx[idx1, idx2])
                img = mnist_test[id]
                axarr[idx1, idx2].imshow(np.reshape(img, [28, 28]))
                axarr[idx1, idx2].set_xticks([])
                axarr[idx1, idx2].set_yticks([])
        plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(log_dir + "/samples_disc/" + str(it) + "_d" + name + ".png")
        plt.close()

    create_image(disc1_idx, "1")


def sample_cont_test_set(encodings, it, num_disc_vars=10, z_dim=16):
    max_c1 = np.zeros(10)
    max_c1_idx = [0] * 10
    min_c1 = np.ones(10)
    min_c1_idx = [0] * 10
    max_c2 = np.zeros(10)
    max_c2_idx = [0] * 10
    min_c2 = np.ones(10)
    min_c2_idx = [1] * 10
    for idx, rep in enumerate(encodings):
        label = np.argmax(rep[z_dim:z_dim + num_disc_vars])
        c1 = rep[-2]
        c2 = rep[-1]
        # if c1 > max_c1[label]:
        if c1 > max_c1[label]:  # and c1 < 2:
            max_c1[label] = c1
            max_c1_idx[label] = idx
        # if c1 < min_c1[label]:
        if c1 < min_c1[label]:  # and c1 > -2:
            min_c1[label] = c1
            min_c1_idx[label] = idx
        # if c2 > max_c2[label]:
        if c2 > max_c2[label]:  # and c2 < 2:
            max_c2[label] = c2
            max_c2_idx[label] = idx
        # if c2 < min_c2[label]:
        if c2 < min_c2[label]:  # and c2 > -2:
            min_c2[label] = c2
            min_c2_idx[label] = idx

    f, axarr = plt.subplots(2, 10)
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

    for idx, c1 in enumerate(max_c1_idx):
        img = mnist_test[c1]
        axarr[0, idx].imshow(np.reshape(img, [28, 28]))
        axarr[0, idx].set_xticks([])
        axarr[0, idx].set_yticks([])
        # scipy.misc.imsave('test_imgs/c1_max_' + str(c1) + 'img_' + str(idx) + '.jpg', np.reshape(img, [28, 28]))
    for idx, c1 in enumerate(min_c1_idx):
        img = mnist_test[c1]
        axarr[1, idx].imshow(np.reshape(img, [28, 28]))
        axarr[1, idx].set_xticks([])
        axarr[1, idx].set_yticks([])
        # scipy.misc.imsave('test_imgs/c1_min_' + str(c1) + 'img_' + str(idx) + '.jpg', np.reshape(img, [28, 28]))
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(log_dir + "/samples_cont/" + str(it) + "_c1.png")
    plt.close()

    f, axarr = plt.subplots(2, 10)
    for idx, c2 in enumerate(max_c2_idx):
        img = mnist_test[c2]
        axarr[0, idx].imshow(np.reshape(img, [28, 28]))
        axarr[0, idx].set_xticks([])
        axarr[0, idx].set_yticks([])
        # scipy.misc.imsave('test_imgs/c2_max_' + str(c2) + 'img_' + str(idx) + '.jpg', np.reshape(img, [28, 28]))
    for idx, c2 in enumerate(min_c2_idx):
        img = mnist_test[c2]
        axarr[1, idx].imshow(np.reshape(img, [28, 28]))
        axarr[1, idx].set_xticks([])
        axarr[1, idx].set_yticks([])
        # scipy.misc.imsave('test_imgs/c2_min' + str(c2) + 'img_' + str(idx) + '.jpg', np.reshape(img, [28, 28]))
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(log_dir + "/samples_cont/" + str(it) + "_c2.png")
    plt.close()