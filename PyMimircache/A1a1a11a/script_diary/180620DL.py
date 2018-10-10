
"""


"""

import os, sys, time, json
from collections import deque, defaultdict, Counter
import numpy as np
from PyMimircache.bin.conf import *
from pprint import pprint

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten, Input
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

from tqdm import tqdm
import itertools, functools


sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
from PyMimircache.A1a1a11a.myUtils.DLUtils import get_txt_trace, read_data, split_data, gen_data_binary, gen_data_rand_binary


np.seterr(all="raise")


def NN_compare_keras(dat, batch_size=1024 * 8, feature_type="noColdMiss.noNan", y_type="frd", hidden_layer_size=(128, 32, ), n_epoch=32,
                     diff_thresh=200, steps_per_epoch=8000, **kwargs):

    X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw, X_test_raw, Y_test_raw = split_data(*read_data(dat, feature_type, y_type))
    n_features = X_train_raw.shape[1] * 2
    n_train = X_train_raw.shape[0]

    model_name = "keras_ignore{}_dropout_{}.h5".format(diff_thresh, dat)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    if os.path.exists(model_name):
        model = load_model(model_name)
        print("model {} loaded".format(model_name))
    else:
        model = Sequential()
        model.add(Dense(hidden_layer_size[0], input_dim=n_features, kernel_initializer='normal', activation="relu"))
        for i in range(1, len(hidden_layer_size)):
            model.add(Dropout(0.2))
            model.add(Dense(hidden_layer_size[i], kernel_initializer='normal', activation=keras.activations.relu))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal', activation="sigmoid"))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    print(model.summary())

    history = model.fit_generator(gen_data_rand_binary(X_train_raw, Y_train_raw, batch_size, diff_thresh=diff_thresh),
                                    # steps_per_epoch=n_train**2//batch_size,
                                    steps_per_epoch=steps_per_epoch, epochs=n_epoch,
                                    validation_data=gen_data_rand_binary(X_valid_raw, Y_valid_raw, batch_size, diff_thresh=diff_thresh),
                                  validation_steps=80,
                                  callbacks=[early_stopping],
                                  )
    # valid_accu = model.evaluate_generator(gen_data_rand_binary(X_valid_raw, Y_valid_raw, batch_size), steps=20, max_queue_size=10)
    # print("{} {}".format(model.metrics_names, valid_accu))

    # model.save(model_name)

    # X_test, Y_test = next(gen_data_rand_binary(X_test_raw, Y_test_raw, 16, diff_thresh=diff_thresh))
    # predictions = model.predict(X_test)
    # print(np.hstack((Y_test, predictions)))
    return history



def NN_multiclass(dat, batch_size=-1, feature_type="noColdMiss.noNan", y_type="frd", hidden_layer_size=(128, 32, ), n_epoch=32,
                     diff_thresh=1000, **kwargs):

    X_raw, Y_raw = read_data(dat, feature_type, y_type)
    Y_raw = Y_raw - diff_thresh
    Y_raw[Y_raw <= 0] = 1
    Y_raw = to_categorical(np.log10(Y_raw).astype(np.int))
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = split_data(X_raw, Y_raw)

    n_features = X_train.shape[1]
    n_class = Y_train.shape[1]
    print("shape {} {} {} {} {} {}".format(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape))


    model_name = "keras_multiclass_ignore{}_{}.h5".format(diff_thresh, dat)

    def my_accu(y_true, y_pred):
        return 1 - 0.5 * K.cast(K.abs(K.argmax(y_true) - K.argmax(y_pred)), K.floatx())


    if os.path.exists(model_name):
        model = load_model(model_name)
        print("model {} loaded".format(model_name))
    else:
        model = Sequential()
        model.add(Dense(hidden_layer_size[0], input_dim=n_features, kernel_initializer='normal', activation="relu"))
        for i in range(1, len(hidden_layer_size)):
            model.add(Dropout(0.2))
            model.add(Dense(hidden_layer_size[i], kernel_initializer='normal', activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(n_class, kernel_initializer='normal', activation="softmax"))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=[keras.metrics.categorical_accuracy, my_accu],
                      # metrics=['accuracy', my_accu]
                      )

    print(model.summary())

    if batch_size == -1:
        batch_size = X_train.shape[0]

    history = model.fit(X_train, Y_train, batch_size, epochs=n_epoch, validation_data=(X_valid, Y_valid))
    # model.save(model_name)

    # predictions = model.predict(X_test)
    # print(np.hstack((Y_test, predictions)))
    return history


def NN_regression(dat, batch_size=-1, feature_type="noColdMiss.noNan", y_type="frd", hidden_layer_size=(128, 32, ), n_epoch=32,
                     diff_thresh=200, steps_per_epoch=8000, **kwargs):

    X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw, X_test_raw, Y_test_raw = read_data(dat, feature_type, y_type)
    n_features = X_train_raw.shape[1]
    n_train = X_train_raw.shape[0]
    n_valid = X_valid_raw.shape[0]
    if batch_size == -1:
        batch_size = X_train_raw.shape[0]

    model_name = "keras_regression_ignore{}_{}.h5".format(diff_thresh, dat)

    if os.path.exists(model_name):
        model = load_model(model_name)
        print("model {} loaded".format(model_name))
    else:
        model = Sequential()
        model.add(Dense(hidden_layer_size[0], input_dim=n_features, kernel_initializer='normal', activation="relu"))
        for i in range(1, len(hidden_layer_size)):
            model.add(Dense(hidden_layer_size[i], kernel_initializer='normal', activation="relu"))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      # loss='mean_absolute_percentage_error',
                      # metrics=['accuracy']
                      )

    print(model.summary())

    X_train = X_train_raw
    Y_train = np.array(Y_train_raw) - diff_thresh
    Y_train[Y_train <= 0] = 1
    Y_train = np.log10(Y_train)

    X_valid = X_valid_raw
    Y_valid = np.array(Y_valid_raw) - diff_thresh
    Y_valid[Y_valid <= 0] = 1
    Y_valid = np.log10(Y_valid)

    rnd = np.random.randint(0, X_test_raw.shape[0], size=16)
    X_test = X_test_raw[rnd, :]
    Y_test = np.array(Y_test_raw[rnd]) - diff_thresh
    Y_test[Y_test <= 0] = 1
    Y_test = np.log10(Y_test)

    history = model.fit(X_train, Y_train, batch_size,
                                  epochs=n_epoch,
                                  validation_data=(X_valid, Y_valid),
                                  )

    model.save(model_name)

    predictions = model.predict(X_test)
    print(np.hstack((Y_test, predictions)))
    return history


def mjolnir_run():
    result = {}
    if os.path.exists("simpleNNwithDropout.result"):
        with open("simpleNNwithDropout.result", "r") as ifile:
            result = json.load(ifile)

    for i in range(106, 1, -1):
        if "w{}".format(i) in result:
            print("w{} ignore".format(i))
            continue

        history = NN_compare_keras("w" + str(i), n_epoch=32)
        result["w" + str(i)] = history.history
        with open("simpleNNwithDropout.result", "w") as ofile:
            json.dump(result, ofile)
        print("w{} done".format(i))

def mjolnir_result():
    from pprint import pprint
    with open("simpleNN.result", "r") as ifile:
        result = json.load(ifile)
    for dat, r in sorted(result.items()):
        print("{}: {:.2f}, {:.2f}".format(dat, r["acc"][-1], r["val_acc"][-1]))
    # pprint(result)

if __name__ == "__main__":
    # mjolnir_run()
    # mjolnir_result()

    with tf.device("/gpu:1"):
        # NN_regression("small", n_epoch=32)
        NN_multiclass("small")
        # NN_multiclass("w106")
        # NN_multiclass("w92")
        # NN_compare_keras("small", n_epoch=32)
        # NN_compare_keras("w92", n_epoch=1)
    # NN_compare_keras("w41")
    # NN_compare_keras()
