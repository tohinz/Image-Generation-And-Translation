import tensorflow as tf

# image logging, discrete variable
def summary_cat(gen_imgs, disc_classes=[10], rows=10, cols=10):
    summary_ops_disc = []
    for idx in range(len(disc_classes)):
        stacked_img = []
        for row in xrange(rows):
            row_img = []
            for col in xrange(cols):
                row_img.append(gen_imgs[(row * rows) + col, :, :, :])
            stacked_img.append(tf.concat(row_img, 1))
        imgs = tf.concat(stacked_img, 0)
        imgs = tf.expand_dims(imgs, 0)
        summary_ops_disc.append(tf.summary.image("images_categorical_" + str(idx), imgs))
    return summary_ops_disc

# image logging, continuous variable
def summary_cont(gen_imgs, num_cont_vars=2, rows=10, cols=10):
    summary_ops_cont = []
    for idx in range(num_cont_vars):
        stacked_img = []
        for row in xrange(rows - 5):
            row_img = []
            for col in xrange(cols):
                row_img.append(gen_imgs[(row * rows) + col, :, :, :])
            stacked_img.append(tf.concat(row_img, 1))
        imgs = tf.concat(stacked_img, 0)
        imgs = tf.expand_dims(imgs, 0)
        summary_ops_cont.append(tf.summary.image("images_continuous_" + str(idx), imgs))
    return summary_ops_cont

# image logging, image reconstruction
def summary_reconstruction(real_imgs, recons_imgs, rows=10, cols=10):
    stacked_img = []
    for row in xrange(rows - 5):
        row_img = []
        for col in xrange(cols):
            row_img.append(real_imgs[(row * rows) + col, :, :, :])
            row_img.append(recons_imgs[(row * rows) + col, :, :, :])
        stacked_img.append(tf.concat(row_img, 1))
    imgs = tf.concat(stacked_img, 0)
    imgs = tf.expand_dims(imgs, 0)
    summary_ops_reconstruction = tf.summary.image("images_reconstruction", imgs)
    return summary_ops_reconstruction
