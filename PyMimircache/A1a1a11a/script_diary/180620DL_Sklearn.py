# coding=utf-8

"""
    machine learning methods (non-NN) for predicting ranking of forward reuse distance


"""

import os, sys, time, random
sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
from PyMimircache.A1a1a11a.myUtils.DLUtils import get_txt_trace, read_data, split_data, gen_data_binary, gen_data_rand_binary, gen_data_regression

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.svm import SVC, LinearSVC, NuSVC

import sklearn as sk
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


def ml_test_classification(dat="small", **kwargs):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = split_data(read_data(dat, "noColdMiss.noNan", "frd"))
    clf, PARTIAL = SGDClassifier(), True
    clf, PARTIAL = MLPClassifier(), True
    clf, PARTIAL = RandomForestClassifier(n_jobs=-1, verbose=2), False

    # The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples
    # clf, PARTIAL = SVC(verbose=2), False

    # clf, PARTIAL = NuSVC(verbose=2), False
    # clf, PARTIAL = LinearSVC(dual=False, verbose=2), False
    # print(clf)

    if PARTIAL:
        data_gen = gen_data_rand_binary(X_train, Y_train, batch_size=2000)
        X, Y = next(data_gen)
        clf.partial_fit(X, Y.ravel(), classes=(0, 1))
        for i in range(20000):
            try:
                X, Y = next(data_gen)
                # print("{} {} {} {}".format(i, X.shape, Y.shape, Y.ravel().shape))
                clf.partial_fit(X, Y.ravel())
            except Exception as e:
                print(e)


            if i % 200 == 0:
                X, Y = next(data_gen)
                Y_pred = clf.predict(X)
                print(sk.metrics.accuracy_score(Y, Y_pred))
    else:
        train_batch_size = kwargs.get("train_batch_size", 20000000)
        valid_batch_size = kwargs.get("valid_batch_size", 2000000)
        data_gen_train = gen_data_rand_binary(X_train, Y_train, batch_size=train_batch_size)
        data_gen_valid = gen_data_rand_binary(X_valid, Y_valid, batch_size=valid_batch_size)
        X, Y = next(data_gen_train)
        clf.fit(X, Y.ravel())

        X, Y = next(data_gen_train)
        Y_pred = clf.predict(X)
        train_accu = sk.metrics.accuracy_score(Y.ravel(), Y_pred)
        print("train accuracy: {}".format(train_accu))

        X, Y = next(data_gen_valid)
        Y_pred = clf.predict(X)
        valid_accu = sk.metrics.accuracy_score(Y.ravel(), Y_pred)
        print("validation accuracy: {}".format(valid_accu))

        print(clf.feature_importances_)

        with open("randomForest", "a") as ofile:
            ofile.write("{} train {:.4f} valid {:.4f}\n".format(dat, train_accu, valid_accu))

    print(clf)
    return clf


def ml_test_regression(dat="small", **kwargs):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = split_data(read_data(dat, "noColdMiss.noNan", "frd"))
    clf, PARTIAL = RandomForestRegressor(n_jobs=-1, verbose=2), False

    # The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples
    # clf, PARTIAL = SVC(verbose=2), False

    # clf, PARTIAL = NuSVC(verbose=2), False
    # clf, PARTIAL = LinearSVC(dual=False, verbose=2), False
    # print(clf)

    train_batch_size = kwargs.get("train_batch_size", 2000000)
    valid_batch_size = kwargs.get("valid_batch_size", 200000)
    data_gen_train = gen_data_regression(X_train, Y_train, batch_size=train_batch_size)
    data_gen_valid = gen_data_regression(X_valid, Y_valid, batch_size=valid_batch_size)
    X, Y = next(data_gen_train)
    clf.fit(X, Y.ravel())

    X, Y = next(data_gen_train)
    # Y_pred = clf.predict(X)
    # train_accu = sk.metrics.accuracy_score(Y.ravel(), Y_pred)
    train_accu = clf.score(X, Y)
    print("train accuracy: {}".format(train_accu))

    X, Y = next(data_gen_valid)
    # Y_pred = clf.predict(X)
    # valid_accu = sk.metrics.accuracy_score(Y.ravel(), Y_pred)
    valid_accu = clf.score(X, Y)
    print("validation accuracy: {}".format(valid_accu))

    print(clf.feature_importances_)

    with open("randomForestRegression", "a") as ofile:
        ofile.write("{} train {:.4f} valid {:.4f}\n".format(dat, train_accu, valid_accu))

    print(clf)


if __name__ == "__main__":
    # ml_test_classification()
    ml_test_regression()

    # for i in range(106, 0, -1):
    #     ml_test_classification(dat="w{}".format(i))