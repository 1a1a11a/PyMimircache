# coding=utf-8

import os, sys, time, random, socket
import numpy as np

from PyMimircache.bin.conf import get_reader

PATH_PREFIX = ""

if "node" in socket.gethostname():
    PATH_PREFIX = "/research/jason/DL/cphy/"

OUTPUT_FOLDER = PATH_PREFIX + "180612Features"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def get_txt_trace(dat, dat_type):
    if os.path.exists(dat):
        return
    reader = get_reader(dat, dat_type)
    with open(dat, "w") as ofile:
        for i in reader:
            ofile.write("{}\n".format(i))

def read_data(dat, feature_type, y_type, split_ratio=(6,2,2)):
    print("read {} {} {} {}".format(dat, feature_type, y_type, split_ratio))
    y_type_to_col = {"opt_rank": 0, "frd": 1, "frt": 2, "fdist": 3, "no_future_reuse": 4}
    X = np.loadtxt(os.path.join(OUTPUT_FOLDER, "{}.X.{}".format(dat, feature_type)), delimiter=",")
    Y = np.loadtxt(os.path.join(OUTPUT_FOLDER, "{}.Y.noColdMiss".format(dat)), delimiter=",")
    Y = Y.astype(np.int64)[:, y_type_to_col[y_type]].reshape(X.shape[0], 1)
    Y[Y==-1] = X.shape[0]

    print("read X shape {} Y shape {}".format(X.shape, Y.shape))

    train_sz, valid_sz, test_sz = [int(X.shape[0] / sum(split_ratio) * i) for i in split_ratio]
    X_train, Y_train = X[:train_sz, :], Y[:train_sz]
    X_valid, Y_valid = X[train_sz : train_sz+valid_sz, :], Y[train_sz : train_sz+valid_sz]
    X_test,  Y_test  = X[train_sz+valid_sz:, :], Y[train_sz+valid_sz:]

    print("X_train {}, Y_train {}, X_valid {}, Y_valid {}, X_test {}, Y_test {}".format(
        X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape
    ))
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def gen_data(X, Y, batch_size, step=1):
    print("use normal data generator")
    n_samples = X.shape[0]
    n_features = X.shape[1] * 2
    batch_X = np.zeros((batch_size, n_features))
    batch_Y = np.zeros((batch_size, 1))
    dim1 = 0
    dim2 = 0
    not_finished = True
    while not_finished:
        for pos in range(batch_size):
            batch_X[pos, :] = X[(dim1, dim2), :].reshape(-1, n_features)
            if Y[dim1] > Y[dim2]:
                batch_Y[pos] = 1
            else:
                batch_Y[pos] = 0


            dim2 += step
            if dim2 >= n_samples:
                dim1 += step
                dim2 = 0
                if dim1 >= n_samples:
                    not_finished = False
                    break
        # print("{} {}".format(dim1, dim2))
        # print(sum(batch_Y))
        yield batch_X, batch_Y


def gen_data_rand(X, Y, batch_size, n_batch=20000):
    print("use rand data generator")
    n_samples = X.shape[0]
    n_features = X.shape[1] * 2
    batch_ind = 0
    while batch_ind < n_batch:
        rand1 = np.random.randint(0, n_samples, size=(batch_size,))
        rand2 = np.random.randint(0, n_samples, size=(batch_size,))

        batch_Y1 = Y[rand1]
        batch_Y2 = Y[rand2]
        keep_pos = (np.abs(batch_Y1 - batch_Y2) > 200).reshape(-1)

        rand1 = rand1[keep_pos]
        rand2 = rand2[keep_pos]
        batch_X1 = X[rand1, :]
        batch_X2 = X[rand2, :]

        # use minus reduce accuracy
        # batch_X = np.subtract(batch_X1, batch_X2)

        batch_X = np.hstack((batch_X1, batch_X2))

        batch_Y1 = Y[rand1]
        batch_Y2 = Y[rand2]
        batch_Y = np.sign(batch_Y1 - batch_Y2)
        batch_Y[batch_Y == -1] = 0
        n_batch += 1

        yield batch_X, batch_Y

def gen_all_data(X, Y):
    n_samples = X.shape[0]
    n_features = X.shape[1] * 2
    print("generate all data, allocate {} * {} matrix".format(n_samples ** 2, n_features))

    all_sub_X = []
    all_sub_Y = []


    t0 = time.time()
    for i in range(n_samples):
        if i % 200 == 0:
            print("{} {} s".format(i, time.time() - t0))
            t0 = time.time()
        keep_pos = (np.abs(Y - Y[i]) > 200).reshape(-1)

        temp = np.tile(X[i, :].reshape(1, -1), (np.sum(keep_pos), 1))
        X1 = np.hstack((temp, X[keep_pos, :]))
        all_sub_X.append(X1)

        Y1 = np.sign(Y - Y[i])
        Y1[Y1 == -1] = 0
        Y1 = Y1[keep_pos]
        all_sub_Y.append(Y1)

        if i % 200 == 0:
            t1 = time.time()
            all_X = np.vstack(all_sub_X)
            all_Y = np.vstack(all_sub_Y)
            print("{} x shape {} y shape {} {}s".format(i, all_X.shape, all_Y.shape, time.time() - t1))

    all_X = np.vstack(all_sub_X)
    all_Y = np.vstack(all_sub_Y)

    return all_X, all_Y


def test():
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = read_data("small", "noColdMiss.noNan", "frd")
    gen_all_data(X_train, Y_train)

if __name__ == "__main__":
    print("DLUtils")