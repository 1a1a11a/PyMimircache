# coding=utf-8
import os, sys
from collections import defaultdict
from multiprocessing import Process, Pool, cpu_count
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

TRAIN_RATIO = 0.3
CUTOFF = 7




def non_cv_specific(dat=None, reg=RandomForestClassifier(n_jobs=1)):
    my_data = np.genfromtxt(dat, delimiter=',')
    cutoff = len(my_data)//CUTOFF

    y = my_data[cutoff:-cutoff, 0].astype(int)
    x = my_data[cutoff:-cutoff, 1:] # [i for i in range(1, my_data.shape[1]) if i!=3 and i!=4 and i!=5 and i!=6]]

    # treat -1 as maxY+1
    # maxY = max(y)
    # y[y==-1] = maxY + 1



    n = len(y)
    train_size = int(n * TRAIN_RATIO)
    train_x = x[: train_size]
    train_y = y[: train_size]



    filter_out = np.logical_not(np.logical_or( train_x[:,0] == -1, train_y[:] == -1) )
    train_x = train_x[filter_out]
    train_y = train_y[filter_out]


    val_x = x[train_size:]
    val_y = y[train_size:]

    # reg = LinearRegression(normalize=True)
    # reg = LogisticRegression(n_jobs=-1) #, multi_class='multinomial', solver="newton-cg", max_iter=1000)
    # reg = MultinomialNB()
    # reg = RandomForestClassifier(n_jobs=4)
    # reg = SVC(cache_size=16000)
    # reg = LinearSVC()
    # reg = make_pipeline(PolynomialFeatures(degree=2), MultinomialNB())
    # reg = make_pipeline(PolynomialFeatures(degree=2), k)

    # reg = Ridge(normalize=True)
    try:
        reg.fit(train_x, train_y)
        pred_y = reg.predict(val_x)
    except Exception as e:
        print(e)
        return
    # correct = pred_y == val_y


    # valy_m1 = val_y[:] - 1
    # valy_p1 = val_y[:] + 1
    # correct = np.logical_or( (pred_y == val_y), (pred_y == valy_p1) )
    # correct = np.logical_or( (pred_y == valy_m1), correct )
    maxY = np.max(val_y)
    correct = 0
    for y_p, y_t in zip(pred_y, val_y):
        if y_t == -1:
            if y_p >= maxY:
                correct += 1
        else:
            if abs(y_p - y_t) <= 1:
                correct += 1

    accuracy = correct/len(pred_y)

    # print("coef: {}".format(reg.coef_))
    # print("intercept {}".format(reg.intercept_))
    # for x in zip(pred_y, val_y):
    #     if x[0] != x[1]:
    #         print(x)
    print("{} true value: {}".format(dat, countClass(val_y)))
    print("{} predicted value: {}".format(dat, countClass(pred_y)))

    print("{}(specific): {}, {}, +-1 accuracy: {}".format(dat, reg.score(train_x, train_y),
                                                      reg.score(val_x, val_y), accuracy))


def countClass(c):
    d = defaultdict(int)
    for x in c:
        d[int(x)] += 1
    return d


def cv(n_folds=10):
    my_data = np.genfromtxt(DATA_PATH, delimiter=',', max_rows=3000000)
    # enc = OneHotEncoder(sparse=False)
    y = my_data[:, 0]
    x = my_data[:, 1:]
    o_y = y.copy()
    # y[o_y > 10] = 1
    # y[o_y < 10] = 0
    print(np.unique(y))
    print(x.shape)

    n = len(y)
    skf = KFold(n, n_folds=n_folds)
    i = 0
    for train_index, val_index in skf:
        i += 1
        train_x = x[train_index]
        train_y = y[train_index]
        val_x = x[val_index]
        val_y = y[val_index]
        reg = Ridge(normalize=True)
        reg.fit(train_x, train_y)
        print('at {} fold'.format(i))
        print(reg.coef_)
        print(reg.score(train_x, train_y))
        print(reg.score(val_x, val_y))


if __name__ == '__main__':
    feature_set = "features/specific"
    trace_num = "w90"


    ALL_CLASSIFIERS = {LinearRegression(normalize=True): "linearRegression",
                       LogisticRegression(): "logisticRegression",
                       MultinomialNB(): "MNB",
                       RandomForestClassifier(n_jobs=4): "randomForest"}
    for k,v in ALL_CLASSIFIERS.items():
        print(v)
        # continue
        if 'randomForest' in v:
            non_cv_specific('{}/{}_t2000.csv'.format(feature_set, trace_num), reg=k)


    # non_cv_interval('{}/{}_t2000.csv'.format("features/interval", trace_num),
    #                 reg=LinearRegression(normalize=True))
    sys.exit(0)



    ################################## BATCH JOB: cal correlation coefficient ##################################

    with Pool(4) as p:
        p.map(non_cv_specific,
              ["{}/{}".format("features/specific", f)
                           for f in os.listdir("features/specific")])
        # p.map(non_cv_interval,
        #       ["{}/{}".format("features/interval", f)
        #                    for f in os.listdir("features/interval")])

