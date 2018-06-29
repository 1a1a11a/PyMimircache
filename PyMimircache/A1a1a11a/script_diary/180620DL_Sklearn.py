# coding=utf-8

"""
    machine learning methods (non-NN) for predicting ranking of forward reuse distance


"""

import os, sys, time, random
sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
from PyMimircache.A1a1a11a.myUtils.DLUtils import get_txt_trace, read_data, gen_data, gen_data_rand

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

import sklearn as sk
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


def ml_test():
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = read_data("w106", "noColdMiss.noNan", "frd")
    clf, PARTIAL = SGDClassifier(), True
    clf, PARTIAL = MLPClassifier(), True
    clf, PARTIAL = RandomForestClassifier(n_jobs=-1, verbose=2), False

    # The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples
    # clf, PARTIAL = SVC(verbose=2), False

    # clf, PARTIAL = NuSVC(verbose=2), False
    # clf, PARTIAL = LinearSVC(dual=False, verbose=2), False
    # print(clf)

    if PARTIAL:
        data_gen = gen_data_rand(X_train, Y_train, batch_size=2000)
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
        data_gen_train = gen_data_rand(X_train, Y_train, batch_size=20000000)
        data_gen_valid = gen_data_rand(X_valid, Y_valid, batch_size=200000)
        X, Y = next(data_gen_train)
        clf.fit(X, Y.ravel())

        X, Y = next(data_gen_train)
        Y_pred = clf.predict(X)
        print("train accuracy: {}".format(sk.metrics.accuracy_score(Y.ravel(), Y_pred)))

        X, Y = next(data_gen_valid)
        Y_pred = clf.predict(X)
        print("validation accuracy: {}".format(sk.metrics.accuracy_score(Y.ravel(), Y_pred)))

    print(clf)

if __name__ == "__main__":
    ml_test()