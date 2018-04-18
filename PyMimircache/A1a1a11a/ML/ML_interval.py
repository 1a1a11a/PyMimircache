# coding=utf-8
from sklearn.linear_model import LinearRegression

import csv
import numpy as np

import sys, time

sys.path.append("../")
sys.path.append("./")
from PyMimircache import *
from PyMimircache.bin.conf import initConf
from PyMimircache.A1a1a11a.ML.featureGenerator import *

################################## Global variable and initialization ###################################

TRACE_TYPE = "cloudphysics"
TRACE_DIR, NUM_OF_THREADS = initConf(TRACE_TYPE, trace_format='variable')
CUTOFF = 7
TRAIN_RATIO = 0.3

t_virtual = 2000  # 60 * 1000000
t_real = 10 * 1000000


def get_feature_interval(dat, time_interval=60*1000000, outfile="feature_interval.csv"):
    reader = VscsiReader(dat)
    bp = get_breakpoints(reader, 'r', time_interval=time_interval)
    # bp = cHeatmap().gen_breakpoints(reader, 'v', time_interval=time_interval)
    rds = get_rd_interval(reader, bp)
    frds = get_rd_interval(reader, bp, frd_flag=True)
    # selc = get_frd_category(reader, bp, cache_size=CACHE_SIZE)
    rd_ave = get_ave_rd_interval(reader, bp)
    frd_ave = get_ave_frd_interval(reader, bp)
    freq = get_freq_interval(reader, bp)

    req_rate = get_request_rate_interval(reader, bp)
    cold_rate = get_cold_miss_rate_interval(reader, bp)

    with open(outfile, 'w') as ofile:
        writer = csv.writer(ofile)
        for i in range(len(rds)-1):
            # cold miss count, ave frd, then 10 frd_distr
            rd_frd_row = [frd_ave[i][0], frd_ave[i][1]]
            rd_frd_row.extend(["{:.2f}".format(frds[i][j]) for j in range(0, 10)])
            rd_frd_row.extend([rd_ave[i][0], rd_ave[i][1]])
            rd_frd_row.extend(["{:.2f}".format(rds[i][j]) for j in range(0, 10)])

            # freq, req_rate, cold miss rate
            rd_frd_row.extend(["{:.6f}".format(x) for x in (freq[i], req_rate[i], cold_rate[i])])
            # difference of freq, req_rate, cold miss rate
            if i!=0:
                rd_frd_row.extend(["{:.6f}".format(x) for x in (freq[i]-freq[i-1],
                                                            req_rate[i]-req_rate[i-1],
                                                            cold_rate[i]-cold_rate[i-1])])
            else:
                rd_frd_row.extend((0, 0, 1))
            # total 2 + 10 + 2 + 10 + 3 + 3
            writer.writerow(rd_frd_row)



def non_cv_interval(dat=None, y_index=0, reg=LinearRegression(normalize=True)):
    my_data = np.genfromtxt(dat, delimiter=',')
    cutoff = len(my_data) // CUTOFF

    # 0 ~ 11(6)
    y = my_data[cutoff:-cutoff, y_index]
    x = my_data[cutoff:-cutoff, 12:]

    # print(np.unique(y))

    # for i in range(x.shape[-1]):
    #     col = x[:, i]
    #     is_minus = (col == -1).astype(float)
    #     col[col == -1] = 0.  # np.nan
    #     x = np.hstack([x, is_minus.reshape(-1, 1)])

    n = len(y)
    train_size = int(n * TRAIN_RATIO)
    train_x = x[: train_size]
    train_y = y[: train_size]

    val_x = x[train_size:]
    val_y = y[train_size:]

    # if sum(val_y)/len(val_y) <

    # reg = LinearRegression(normalize=True)
    # reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(normalize=True))

    # reg = Ridge(normalize=True)
    reg.fit(train_x, train_y)
    pred_y = reg.predict(val_x)


    ################################ accuracy report #################################
    # test_accuracy = [1, 2, 3, 5, 10, 20]
    # accuracy = {}
    # accuracy_sum = 0
    # for threshold in test_accuracy:
    #     counter = 0
    #     for x in zip(pred_y, val_y):
    #         if abs(x[0] - x[1]) > threshold:
    #             counter += 1
    #             # print(x)
    #     accuracy[threshold] = 1 - counter / len(pred_y)
    #     accuracy_sum += 1 - counter / len(pred_y)
    #
    #
    # if accuracy_sum < 0.02:
    #     return
    # if abs(accuracy[1] + accuracy[2] - 2) < 0.01:
    #     return
    #
    # print("{}_{}(interval): {:.4f}, {:.4e}: {:.2f}(1), {:.2f}(2), {:.2f}(3), {:.2f}(5), "
    #       "{:.2f}(10), {:.2f}(20)".format(dat, y_index, reg.score(train_x, train_y),
    #                                       reg.score(val_x, val_y), accuracy[1], accuracy[2],
    #                                       accuracy[3], accuracy[5], accuracy[10], accuracy[20]))


    ################################ accuracy report #################################
    counter = 0
    if sum(val_y) < 0.01:
        return
    for x in zip(pred_y, val_y):
        if 0.5<x[0]/x[1]<2:
            counter += 1
    accuracy = 1 - counter / len(pred_y)
    print("{}_{}(interval): {:.4f}, {:.4e}: {:.2f}(0.5:2)".format(dat, y_index, reg.score(train_x, train_y),
                                          reg.score(val_x, val_y), accuracy))


################################## BATCH JOB: generate features #################################

def batch_feature_extraction():
    ################ multiprocessing #################
    # with ProcessPoolExecutor(1) as p:
    #     futures = {p.submit(get_feature_interval,
    #                         "{}/{}".format(TRACE_DIR, f),
    #                         t_real,
    #                         "features/interval/{}_tr{}.csv".format(f[:f.find('_')], t_real)):
    #                    f for f in os.listdir(TRACE_DIR) if f.endswith("vscsitrace")}
    #     for future in as_completed(futures):
    #         print(futures[future])

    ################ sequential #######################
    for f in os.listdir(TRACE_DIR):
        if f.endswith("vscsitrace"):
            get_feature_interval("{}/{}".format(TRACE_DIR, f), t_real,
                                "features/interval/{}_tr{}.csv".format(f[:f.find('_')], t_real))
            print(f)



def batch_calAccuracy():
    INPUT_FOLDER = "features/interval/old/"
    for f in os.listdir(INPUT_FOLDER):
        dat = "{}/{}".format(INPUT_FOLDER, f)
        for i in range(2, 12):
            non_cv_interval(dat, y_index=i)
        print("*" * 160)




def test_func():
    dat = "../../data/trace.vscsi"
    dat = "/root/disk2/ALL_DATA/traces/w102_vscsi1.vscsitrace"
    trace_num = dat[dat.rfind('/')+1:dat.rfind('.')]
    ################################ temporary feature extraction functions ##################################
    # trace_num = "w101"
    # dat = "{}/{}_vscsi1.vscsitrace".format(TRACE_DIR, trace_num)

    t1 = time.time()
    get_feature_interval(dat, time_interval=t_real,
                         outfile="features/interval/{}_t{}.csv".format(trace_num, t_real))
    print(time.time() - t1)
    sys.exit(0)

    ################################ temporary cal accuracy functions ##################################
    dat = "features/interval/w{}_tr2000.csv".format(106)

    # for i in range(0, 12):
    #     non_cv_interval(dat, y_index=i)





###################################### main funcitons ########################################
if __name__ == "__main__":
    test_func()
    # batch_calAccuracy()
    # batch_feature_extraction()

