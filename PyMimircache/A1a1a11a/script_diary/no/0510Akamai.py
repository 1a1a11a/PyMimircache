# coding=utf-8
"""
this module tries to characterize the traffic between second layer cache to end user and 
tiered distribution traffic 
"""

import os, sys
sys.path.insert(0, "../../")
sys.path.insert(0, "../../../")
from PyMimircache import *
from PyMimircache.bin.conf import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

TRACE_DIR, NUM_OF_THREADS = initConf(trace_type="Akamai2", trace_format="csv")





def traffic_variation(dat, mode="r", time_interval=-1, **kwargs):
    """
    this function plots the the percentage traffic variation of end-user and non-enduser VS time 
    :param dat:  input data file 
    :param mdoe: real time or virtual time 
    :return: 
    """
    VIRTUAL_TIME_INTERVAL = 100000
    REAL_TIME_INTERVAL = 60
    if time_interval == -1:
        if mode == 'r':
            time_interval = REAL_TIME_INTERVAL
        elif mode == 'v':
            time_interval = VIRTUAL_TIME_INTERVAL

    count_enduser = 0
    count_nonenduser = 0
    last_cut_time = 0
    l = []
    with open(dat, 'r') as ifile:
        for line in ifile:
            line_split = line.split("\t")
            if last_cut_time == 0:
                last_cut_time = float(line_split[0])

            traffic_type = int(line_split[-1])
            if traffic_type == 1:
                count_enduser +=1
            else:
                count_nonenduser += 1

            if mode == 'r':
                current_time = float(line_split[0])
                if current_time - last_cut_time > time_interval:
                    l.append(count_enduser/(count_nonenduser+count_enduser))
                    last_cut_time = current_time
                    count_enduser = 0
                    count_nonenduser = 0

            elif mode == 'v':
                if count_enduser + count_nonenduser == time_interval:
                    l.append(count_enduser/time_interval)
                    count_enduser = 0
                    count_nonenduser = 0
            else:
                print("unknow mode {}".format(mode))
                return

    plt.plot(l)
    plt.xlabel("time ({})".format(mode))
    plt.ylabel("percent from end user")
    plt.savefig("0510fig/{}_trafficDistr_{}.png".format(os.path.basename(dat), mode))
    plt.clf()




############################ RUNNABLE ############################

def run_all_single_thread():
    for f in os.listdir(TRACE_DIR):
        if not os.path.isfile("{}/{}".format(TRACE_DIR, f)):
            continue
        print(f)
        traffic_variation("{}/{}".format(TRACE_DIR, f), "r")
        traffic_variation("{}/{}".format(TRACE_DIR, f), "v")



if __name__ == "__main__":
    run_all_single_thread()