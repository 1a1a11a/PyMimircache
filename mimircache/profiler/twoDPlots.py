import os
import pickle
from collections import deque
from multiprocessing import Array, Process, Queue

import matplotlib
import matplotlib.ticker as ticker
import mimircache.c_heatmap as c_heatmap
import numpy as np
from matplotlib import pyplot as plt

from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.utils.printing import *
from mimircache.profiler.cHeatmap import cHeatmap


def request_num_2d(reader, mode, time_interval, figname="request_num.png"):
    assert mode == 'r' or mode == 'v', "currently only support mode r and v, what mode are you using?"
    break_points = cHeatmap().gen_breakpoints(reader, mode, time_interval)

    l = []
    for i in range(1, len(break_points)):
        l.append(break_points[i] - break_points[i - 1])
    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
    draw2d(l, figname=figname, xticks=xticks, xlabel='time({})'.format(mode),
           ylabel='request num count(interval={})'.format(time_interval),
           title='request num count 2D plot')


def cold_miss_2d(reader, mode, time_interval, figname="cold_miss2d.png"):
    assert mode == 'r' or mode == 'v', "currently only support mode r and v, what mode are you using?"
    break_points = cHeatmap().gen_breakpoints(reader, mode, time_interval)

    cold_miss_list = [0] * (len(break_points) - 1)
    seen_set = set()
    never_see = 0
    for i in range(len(break_points) - 1):
        never_see = 0
        for j in range(break_points[i + 1] - break_points[i]):
            r = next(reader)
            if r not in seen_set:
                seen_set.add(r)
                never_see += 1
        cold_miss_list[i] = never_see

    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
    draw2d(cold_miss_list, figname=figname, xticks=xticks, xlabel='time({})'.format(mode),
           ylabel='cold miss count(interval={})'.format(time_interval),
           title='cold miss count 2D plot')
    reader.reset()


def draw2d(l, **kwargs):
    if 'figname' in kwargs:
        filename = kwargs['figname']
    else:
        filename = '2d_plot.png'

    plt.plot(l)

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    if 'xticks' in kwargs:
        plt.gca().xaxis.set_major_formatter(kwargs['xticks'])
    if 'yticks' in kwargs:
        plt.gca().yaxis.set_major_formatter(kwargs['yticks'])
    if 'title' in kwargs:
        plt.title(kwargs['title'])

    plt.savefig(filename, dpi=600)
    # plt.show()
    colorfulPrint("red", "plot is saved at the same directory")
    plt.clf()






if __name__ == "__main__":
    pass
    # TIME_INTERVAL = 10000000000
    # from mimircache.cacheReader.vscsiReader import vscsiCacheReader
    # PATH = '/run/shm/traces/'
    #
    # reader = vscsiCacheReader('../data/trace.vscsi')
    # for f in os.listdir(PATH):
    #     if f.endswith('vscsitrace'):
    #         reader = vscsiCacheReader(PATH + f)
    #         request_num_2d(reader, 'r', TIME_INTERVAL,
    #                        figname='0627_request_num/'+f+'_request_num_'+str(TIME_INTERVAL)+'.png')
    # request_num_2d(reader, 'r', 10000000)
    # cold_miss_2d(reader, 'r', 10000000)
