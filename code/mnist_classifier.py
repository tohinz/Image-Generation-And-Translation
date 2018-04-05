from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing import image

# activation functions
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x), alpha)

def relu_batch_norm(x):
    return tf.nn.relu(tcl.batch_norm(x))


def load_mnist_data():
    mnist = input_data.read_data_sets('./MNIST_data', validation_size=5000, one_hot=True)

    x_train = mnist.train.images
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    y_train = mnist.train.labels

    x_val = mnist.validation.images
    x_val = np.reshape(x_val, (-1, 28, 28, 1))
    y_val = mnist.validation.labels

    x_test = mnist.test.images
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    y_test = mnist.test.labels

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(input_shape = (28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(4, 4), input_shape=input_shape, padding="same", strides=2,
                     kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(4, 4), input_shape=input_shape, padding="same", strides=2,
                     kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(4, 4), input_shape=input_shape, padding="same", strides=2,
                     kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(4, 4), input_shape=input_shape, padding="same", strides=2,
                     kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


# training function
def train_model():
    train_data, val_data, test_data = load_mnist_data()

    model = build_model()

    save_model = keras.callbacks.ModelCheckpoint("weights.epoch-{epoch:02d}.val_acc-{val_acc:.4f}.hdf5",
                                                   monitor='val_acc', verbose=0, save_best_only=True,
                                                   save_weights_only=False, mode='auto', period=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True,
                                              write_grads=False, write_images=False, embeddings_freq=0,
                                              embeddings_layer_names=None, embeddings_metadata=None)

    model.fit(train_data[0], train_data[1],
              batch_size=64,
              epochs=5,
              verbose=1,
              validation_data=val_data,
              callbacks=[early_stopping, save_model, tensorboard])

    score = model.evaluate(test_data[0], test_data[1], verbose=1)
    print('Test loss: {.4f}'.format(score[0]))
    print('Test accuracy: {.4f}'.format(score[1]))


def train_classifier():
    mnist = input_data.read_data_sets('./MNIST_data', validation_size=0, one_hot=True)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 28,28,1])
    dropout_keep = tf.placeholder(tf.float32)

    conv1 = tcl.conv2d(
        x, 64, [4, 4], [2, 2],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        activation_fn=tf.identity)
    conv1 = leaky_relu_batch_norm(conv1)
    conv1 = tcl.max_pool2d(conv1, kernel_size = [2, 2], stride = [1, 1])
    conv1 = tf.nn.dropout(conv1, keep_prob = dropout_keep)
    print(conv1)
    conv2 = tcl.conv2d(
        conv1, 128, [4, 4], [2, 2],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        activation_fn=tf.identity)
    conv2 = leaky_relu_batch_norm(conv2)
    conv2 = tcl.max_pool2d(conv2, kernel_size = [2, 2], stride = [1, 1])
    conv2 = tf.nn.dropout(conv2, keep_prob = dropout_keep)
    print(conv2)
    conv3 = tcl.conv2d(conv2, 256, [4, 4], [2, 2],
        weights_initializer = tf.random_normal_initializer(stddev = 0.01),
        activation_fn = tf.identity)
    conv3 = leaky_relu_batch_norm(conv3)
    conv3 = tcl.max_pool2d(conv3, kernel_size = [2, 2], stride = [1, 1])
    conv3 = tf.nn.dropout(conv3, keep_prob = dropout_keep)
    print(conv3)
    conv4 = tcl.conv2d(conv3, 256, [4, 4], [2, 2],
        weights_initializer = tf.random_normal_initializer(stddev = 0.01),
        activation_fn = tf.identity)
    conv4 = leaky_relu_batch_norm(conv4)
    print(conv4)
    exit()
    conv4 = tcl.flatten(conv4)
    fc1 = tcl.fully_connected(
        conv4, 1024,
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        activation_fn=tf.identity)
    fc1 = leaky_relu_batch_norm(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob = dropout_keep)
    logits_ = tcl.fully_connected(fc1, 10, activation_fn = None)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=logits_)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits_,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())

    test_set_acc = 0.0

    for i in range(50000):
      batch = mnist.train.next_batch(50)
      if i % 1000 == 0:
        print(i)
        _test_set_acc = accuracy.eval(feed_dict={x: np.reshape(mnist.test.images, [-1, 28, 28, 1]), y_: mnist.test.labels, dropout_keep: 1.0})
        print("test set accuracy %g"%_test_set_acc)
        if _test_set_acc > test_set_acc:
            test_set_acc = _test_set_acc
            saver.save(sess, 'mnist_classifier/mnist_clf_model_'+str(test_set_acc)+'.ckpt')
        # for xb, yb in zip(xs, ys):
        #   print("accuracy %g"%accuracy.eval(feed_dict={x: np.reshape(xb, [-1, 28, 28, 1]), y_: yb, dropout_keep: 1.0}))
      train_step.run(feed_dict={x: np.reshape(batch[0], [50, 28, 28, 1]), y_: batch[1], dropout_keep: 0.5})


def normalize_image(img):
    return img*1.0/255.0


def predict(num_samples=500, img_path = "mnist_test_imgs"):
    num_samples = num_samples
    model = keras.models.load_model("code/weights.37-0.9943.hdf5")
    img_generator = image.ImageDataGenerator(preprocessing_function=normalize_image)
    validation_generator = img_generator.flow_from_directory(directory=img_path, target_size=(28,28),
                                                             batch_size=500, shuffle=False, color_mode="grayscale")
    predictions = model.predict_generator(validation_generator, steps=10*num_samples/500)


    real_labels = np.zeros((10*num_samples, 10))
    for idx in range(10):
        real_labels[idx*num_samples:idx*num_samples+num_samples, idx] = 1

    accuracy = np.mean(np.equal(np.argmax(predictions, axis=1), np.argmax(real_labels, axis=1)))
    return accuracy
