# coding=utf-8


import os, sys, time
from collections import deque, defaultdict, Counter
import numpy as np
from PyMimircache.bin.conf import *
from pprint import pprint

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten, Input
from keras.models import Model
from keras.models import load_model

from tqdm import tqdm
import itertools, functools


sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
from PyMimircache.A1a1a11a.myUtils.DLUtils import get_txt_trace, read_data, split_data, gen_data_binary, gen_data_rand_binary
import iris_data


def gen_data_rand2():
    for i in range(80):
        yield i, i**3


def generator():
    # for i in itertools.count(1):
    for i in range(200):
        yield (i, [1] * i)



def mytest1(batch_size=128, feature_type="noColdMiss.noNan", y_type="frd", hidden_layer_size=(32,)):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = read_data("small", feature_type, y_type)
    n_features = X_train.shape[1] * 2

    gen_training   = functools.partial(gen_data_rand_binary, X_train, Y_train, 1024)
    gen_validation = functools.partial(gen_data_rand_binary, X_valid, Y_valid, 1024)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=(tf.int64, tf.int64),
                                             output_shapes=(tf.TensorShape([]), tf.TensorShape([None])))

    dataset = tf.data.Dataset().batch(batch_size).from_generator(gen_training, output_types=(tf.float32, tf.float32),
                                                                 output_shapes=(tf.TensorShape([None, 68]), tf.TensorShape([None, 1]))
                                                                 )

    # value = dataset.make_one_shot_iterator().get_next()
    it = dataset.make_initializable_iterator()
    value = it.get_next()

    with tf.Session() as sess:
        sess.run(it.initializer)
        for i in range(20):
            print(sess.run(value))


def NN_compare(dat, batch_size=1024 * 8, feature_type="noColdMiss.noNan", y_type="frd", hidden_layer_size=(32,), **kwargs):

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = read_data(dat, feature_type, y_type)
    n_features = X_train.shape[1] * 2
    n_classes = 1

    feature_columns = [tf.feature_column.numeric_column("X", shape=[n_features])]
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/jason/")

    def data_gen(X, Y, batch_size):
        # gen_training = functools.partial(gen_data_rand_binary, X_train, Y_train, 1024)
        gen_data = functools.partial(gen_data_rand_binary, X, Y, batch_size)
        # gen_validation = functools.partial(gen_data_rand_binary, X_valid, Y_valid, 1024)
        dataset = tf.data.Dataset().batch(batch_size).from_generator(gen_data,
                                                                     output_types=(tf.float32, tf.float32),
                                                                     output_shapes=(tf.TensorShape([None, n_features]),
                                                                                    tf.TensorShape([None, n_classes]))
                                                                     )

        # value = dataset.make_one_shot_iterator().get_next()
        it = dataset.make_initializable_iterator()
        # tf_x, tf_y = it.get_next()
        return it.get_next()


    def my_model_fn(features, labels, mode, params):
        print(features)

        # Use `input_layer` to apply the feature columns.
        print(features)
        print("{} {}".format(mode, params))
        net = tf.feature_column.input_layer(features, params['feature_columns'])
        # Build the hidden layers, sized according to the 'hidden_units' param.
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        logits = tf.layers.dense(net, params['n_classes'], activation=None)

        # Compute predictions.
        # predicted_classes = tf.argmax(logits, 1)
        predicted_classes = int(tf.sigmoid(logits) + 0.5)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.sigmoid(logits)/2 + 1,
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)


        # Compute loss.
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.log_loss(labels=labels, logits=logits)
        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')

        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        params={
            'feature_columns': feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 1,
        })

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(training_set.data)},
    #     y=np.array(training_set.target),
    #     num_epochs=None,
    #     shuffle=True)


    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    def train_input_fn(features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_next()


    # Train the Model.
    classifier.train(
        input_fn=lambda: data_gen(X_train, Y_train, batch_size),
        # input_fn=lambda:train_input_fn(train_x, train_y, batch_size),
        steps=2000)


    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=lambda: data_gen(X_valid, Y_valid, batch_size))["accuracy"]

    print("\nValidation Accuracy: {0:f}\n".format(accuracy_score))



    sys.exit(1)

    # tf_x = tf.placeholder(tf.float32, shape=[None, n_features])
    # tf_y = tf.placeholder(tf.float32, shape=[None, 1])

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


    data_gen = gen_data_rand_binary(X_train, Y_train, batch_size)

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


if __name__ == "__main__":
    # mytest1()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=mytest1)