
"""


"""

import os, sys, time
from collections import deque, defaultdict, Counter
import numpy as np
from PyMimircache.bin.conf import *
from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
import bisect
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pprint import pprint

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten, Input
from keras.models import Model
from keras.models import load_model

from tqdm import tqdm


sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
from PyMimircache.A1a1a11a.myUtils.DLUtils import get_txt_trace, read_data, gen_data, gen_data_rand


np.seterr(all="raise")


def NN_compare(dat, batch_size=1024 * 8, feature_type="noColdMiss.noNan", y_type="frd", hidden_layer_size=(32,), **kwargs):

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = read_data(dat, feature_type, y_type)
    n_features = X_train.shape[1] * 2

    dataset = tf.data.Dataset().batch(batch_size).from_generator(gen_data_rand,
                                                        output_types=(tf.float32, tf.float32),
                                                        output_shapes=(tf.TensorShape([None, 2])))

    value = dataset.make_initializable_iterator().get_next()

    with tf.Session() as sess:
        for i in range(20):
            print(sess.run(value))

    tf_x = tf.placeholder(tf.float32, shape=[None, n_features])
    tf_y = tf.placeholder(tf.float32, shape=[None, 1])

    tf_weight1 = tf.Variable(tf.random_normal([n_features, hidden_layer_size[0]]))
    tf_bias1 = tf.Variable(tf.random_normal([hidden_layer_size[0]]))
    tf_h1 = tf.nn.relu(tf.matmul(tf_x, tf_weight1) + tf_bias1)

    tf_weight2 = tf.Variable(tf.random_normal([hidden_layer_size[0], 1]))
    tf_bias2 = tf.Variable(tf.random_normal([1]))
    tf_h2 = tf.matmul(tf_h1, tf_weight2) + tf_bias2

    # tf_y_pred = tf.sigmoid(tf_h2)
    # cross_entropy = tf.reduce_mean(
    #         tf_y * - tf.log(tf_y_pred) + (1 - tf_y) * -tf.log(1 - tf_y_pred)
    #     )

    tf_y_pred = tf_h2
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=tf_y_pred))

    # train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1).minimize(cross_entropy)


    data_gen = gen_data_rand(X_train, Y_train, batch_size)

    # return rand1

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        total_loss = 0
        total_accu = 0
        for i in range(200000):
            # gen batch
            # rand1 = np.random.randint(0, X_train.shape[0], size=(batch_size, ))
            # rand2 = np.random.randint(0, X_train.shape[0], size=(batch_size, ))
            # batch_x1 = X_train[rand1, :]
            # batch_x2 = X_train[rand2, :]
            # batch_y = (Y_train[rand1] > Y_train[rand2]).astype(np.float32)
            batch_x, batch_y = next(data_gen)
            loss, _ = sess.run([cross_entropy, train_step], feed_dict={tf_x: batch_x, tf_y: batch_y})
            total_loss += loss

            if i % 200 == 0:
                # batch_x, batch_y = next(data_gen)
                # tf_accu = tf.metrics.accuracy(tf_y, tf_y_pred)

                print("{}".format(total_loss / 200))
                total_loss, total_accu = 0, 0



def NN_compare_keras(dat, batch_size=1024 * 8, feature_type="noColdMiss.noNan", y_type="frd", hidden_layer_size=(32, ), n_epoch=32, **kwargs):

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = read_data(dat, feature_type, y_type)
    n_features = X_train.shape[1] * 2
    n_train = X_train.shape[0]

    model_name = "keras_ignore200_noConcatenate_{}.h5".format(dat)

    if os.path.exists(model_name):
        model = load_model(model_name)
        print("model {} loaded".format(model_name))
    else:
        model = Sequential()
        for i in range(len(hidden_layer_size)):
            model.add(Dense(hidden_layer_size[i], input_dim=n_features, kernel_initializer='normal', activation="relu"))
        model.add(Dense(1, kernel_initializer='normal', activation="sigmoid"))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    print(model.summary())

    for i in range(n_epoch):
        model.fit_generator(gen_data_rand(X_train, Y_train, batch_size),
                            steps_per_epoch=n_train**2//batch_size,
                            # steps_per_epoch=20000,
                            epochs=1)
                            # validation_data=gen_data(X_valid, Y_valid, batch_size), validation_steps=X_valid.shape[0]**2//batch_size)
        model.save(model_name)

    x = model.evaluate_generator(gen_data_rand(X_valid, Y_valid, batch_size), steps=X_valid.shape[0]**2//batch_size)
    print(x)

    predictions = model.predict_generator(gen_data(X_test, Y_test, batch_size), steps=X_test.shape[0]**2//batch_size)


def simple_NN(dat_name, feature_type="noColdMiss.noNan"):
    X = np.loadtxt("{}.X.{}".format(dat_name, feature_type))
    Y = np.loadtxt("{}.Y.noColdMiss".format(dat_name))



if __name__ == "__main__":
    # simple_NN("small")
    # myt()
    # myt2("w60")
    # NN_compare("small")
    NN_compare_keras("small")
    # test2()
    # test()