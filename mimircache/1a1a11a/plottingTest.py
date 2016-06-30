

import os
import pickle
from collections import deque
from multiprocessing import Array, Process, Queue

import matplotlib
import matplotlib.ticker as ticker
import mimircache.c_heatmap as c_heatmap
import numpy as np
from matplotlib import pyplot as plt

from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.utils.printing import *
from mimircache.profiler.cHeatmap import cHeatmap
from mimircache.top.cachecow import cachecow
from mimircache.profiler.twoDPlots import *


def ave_freq_2d(reader, mode, time_interval, figname="ave_freq.png"):
    assert mode == 'r' or mode == 'v', "currently only support mode r and v, what mode are you using?"
    break_points = cHeatmap().gen_breakpoints(reader, mode, time_interval)

    freq_dict = {}
    l = []
    for i in range(1, len(break_points)):
        ave_sec = 0
        ave_freq = 0
        for j in range(break_points[i-1], break_points[i]):
            sec = next(reader)
            if sec in freq_dict:
                ave_freq += freq_dict[sec]
                freq_dict[sec] += 1
            else:
                freq_dict[sec] = 1

        print("{}: {}".format(ave_freq, break_points[i]-break_points[i-1]))
        ave_freq = ave_freq / (break_points[i] - break_points[i-1])
        l.append(ave_freq)
    reader.reset()



    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
    draw2d(l, figname=figname, xticks=xticks, xlabel='time({})'.format(mode),
           ylabel='ave freq(interval={})'.format(time_interval),
           title='ave freq 2D plot')


def ave_sec_2d(reader, mode, time_interval, figname="ave_freq.png"):
    assert mode == 'r' or mode == 'v', "currently only support mode r and v, what mode are you using?"
    break_points = cHeatmap().gen_breakpoints(reader, mode, time_interval)

    freq_dict = {}
    l = []
    for i in range(1, len(break_points)):
        ave_sec = 0
        ave_freq = 0
        for j in range(break_points[i-1], break_points[i]):
            sec = next(reader)
            if sec in freq_dict:
                ave_freq += freq_dict[sec]
                freq_dict[sec] += 1
            else:
                freq_dict[sec] = 1

        print("{}: {}".format(ave_freq, break_points[i]-break_points[i-1]))
        ave_freq = ave_freq / (break_points[i] - break_points[i-1])
        l.append(ave_freq)
    reader.reset()



    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
    draw2d(l, figname=figname, xticks=xticks, xlabel='time({})'.format(mode),
           ylabel='ave freq(interval={})'.format(time_interval),
           title='ave freq 2D plot')



if __name__ == "__main__":
    TIME_INTERVAL = 600000000
    reader = vscsiCacheReader('../data/trace.vscsi')
    # reader = vscsiCacheReader('../data/traces/w38_vscsi1.vscsitrace')
    # ave_freq_2d(reader, 'r', TIME_INTERVAL)

    # cold_miss_2d(reader, 'r', TIME_INTERVAL)

    c = cachecow()
    c.vscsi('../data/trace.vscsi')
    c.differential_heatmap('r', 10000000, "hit_rate_start_time_end_time", algorithm1="ARC", algorithm2="LRU", cache_size=2000, num_of_process=8)