# coding=utf-8


import os, sys, time, pickle, glob
from PyMimircache import *
from PyMimircache.utils.printing import *



def load_global_hash(dat="/home/jason/ALL_DATA/Akamai/global.hash"):
    INFO("begin loading global hash")
    with open(dat, 'rb') as ifile:
        global_hash = pickle.load(ifile)
    INFO("done loading global hash")
    return global_hash


def save_obj_time_dict(inDat="/home/jason/ALL_DATA/Akamai/Akamai.key.sort2",
                       oDir="/home/jason/ALL_DATA/Akamai/obj_time/"):
    global_hash = load_global_hash()

    dict1_10 = {}
    dict11_20 = {}
    dict21_30 = {}
    dict31_40 = {}
    dict41_50 = {}
    dict51_60 = {}
    dict61_70 = {}
    dict71_80 = {}
    dict81_90 = {}
    dict91_100 = {}
    dict101_150 = {}
    dict151_200 = {}
    dict201_300 = {}
    dict301_500 = {}
    dict501_1000 = {}
    dict1001_up = {}

    timestamps = []
    obj = None
    with open(inDat) as ifile:
        for n, line in enumerate(ifile):
            if (len(line.strip()) == 0 or line == "\n"):
                if obj is not None:
                    # one obj done, save to dict
                    if len(timestamps) <= 10:
                        dict1_10[obj] = timestamps
                    elif len(timestamps) <= 20:
                        dict11_20[obj] = timestamps
                    elif len(timestamps) <= 30:
                        dict21_30[obj] = timestamps
                    elif len(timestamps) <= 40:
                        dict31_40[obj] = timestamps
                    elif len(timestamps) <= 50:
                        dict41_50[obj] = timestamps
                    elif len(timestamps) <= 60:
                        dict51_60[obj] = timestamps
                    elif len(timestamps) <= 70:
                        dict61_70[obj] = timestamps
                    elif len(timestamps) <= 80:
                        dict71_80[obj] = timestamps
                    elif len(timestamps) <= 90:
                        dict81_90[obj] = timestamps
                    elif len(timestamps) <= 100:
                        dict91_100[obj] = timestamps
                    elif len(timestamps) <= 150:
                        dict101_150[obj] = timestamps
                    elif len(timestamps) <= 200:
                        dict151_200[obj] = timestamps
                    elif len(timestamps) <= 300:
                        dict201_300[obj] = timestamps
                    elif len(timestamps) <= 500:
                        dict301_500[obj] = timestamps
                    elif len(timestamps) <= 1000:
                        dict501_1000[obj] = timestamps
                    else:
                        dict1001_up[obj] = timestamps
                obj = None
                timestamps = []
            else:
                line_split = line.strip().split("\t")
                # print(line_split)
                if obj is None:
                    if line_split[4] not in global_hash:
                        raise RuntimeError("{} not in global hash".format(line_split[4]))
                    obj = global_hash[line_split[4]]
                timestamps.append(int(line_split[0]))

            if n % 10000000 == 0:
                with open("{}/freq1_10.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict1_10, ofile, protocol=4)
                with open("{}/freq11_20.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict11_20, ofile, protocol=4)
                with open("{}/freq21_30.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict21_30, ofile, protocol=4)
                with open("{}/freq31_40.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict31_40, ofile, protocol=4)
                with open("{}/freq41_50.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict41_50, ofile, protocol=4)
                with open("{}/freq51_60.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict51_60, ofile, protocol=4)
                with open("{}/freq61_70.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict61_70, ofile, protocol=4)
                with open("{}/freq71_80.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict71_80, ofile, protocol=4)
                with open("{}/freq81_90.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict81_90, ofile, protocol=4)
                with open("{}/freq91_100.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict91_100, ofile, protocol=4)
                with open("{}/freq101_150.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict101_150, ofile, protocol=4)
                with open("{}/freq151_200.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict151_200, ofile, protocol=4)
                with open("{}/freq201_300.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict201_300, ofile, protocol=4)
                with open("{}/freq301_500.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict301_500, ofile, protocol=4)
                with open("{}/freq501_1000.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict501_1000, ofile, protocol=4)
                with open("{}/freq1001_up.pickle".format(oDir), 'wb') as ofile:
                    pickle.dump(dict1001_up, ofile, protocol=4)


def mytest():
    load_global_hash()


if __name__ == "__main__":
    # mytest()
    save_obj_time_dict()