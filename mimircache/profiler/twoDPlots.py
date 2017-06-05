# coding=utf-8
"""
this module plots all the two dimensional figures, currently including:
    request number plot
    cold miss plot
    name mapping plot(mapping block to a LBA) for visulization of scan and so on
"""


import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from mimircache.utils.printing import *
from mimircache.profiler.cHeatmap import cHeatmap
import os
import numpy as np


def request_num_2d(reader, mode, time_interval, figname="request_num.png"):
    """
    plot the number of requests per time_interval vs time
    :param reader:
    :param mode: either 'r' or 'v' for real time(wall-clock time) or virtual time(reference time)
    :param time_interval:
    :param figname:
    :return:
    """
    assert mode == 'r' or mode == 'v', "currently only support mode r and v, what mode are you using?"
    break_points = cHeatmap().getBreakpoints(reader, mode, time_interval)

    l = []
    for i in range(1, len(break_points)):
        l.append(break_points[i] - break_points[i - 1])
    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
    draw2d(l, figname=figname, xticks=xticks, xlabel='time({})'.format(mode),
           ylabel='request num count(interval={})'.format(time_interval),
           title='request num count 2D plot')


def cold_miss_count_2d(reader, mode, time_interval, figname="cold_miss_count2d.png"):
    """
    plot the number of cold miss per time_interval
    :param reader:
    :param mode: either 'r' or 'v' for real time(wall-clock time) or virtual time(reference time)
    :param time_interval:
    :param figname:
    :return:
    """
    assert mode == 'r' or mode == 'v', "currently only support mode r and v, what mode are you using?"
    break_points = cHeatmap().getBreakpoints(reader, mode, time_interval)

    cold_miss_list = [0] * (len(break_points) - 1)
    seen_set = set()
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


def cold_miss_ratio_2d(reader, mode, time_interval, figname="cold_miss_ratio2d.png"):
    """
    plot the percent of cold miss per time_interval
    :param reader:
    :param mode: either 'r' or 'v' for real time(wall-clock time) or virtual time(reference time)
    :param time_interval:
    :param figname:
    :return:
    """
    assert mode == 'r' or mode == 'v', "currently only support mode r and v, what mode are you using?"
    break_points = cHeatmap().getBreakpoints(reader, mode, time_interval)

    cold_miss_list = [0] * (len(break_points) - 1)
    seen_set = set()
    for i in range(len(break_points) - 1):
        never_see = 0
        for j in range(break_points[i + 1] - break_points[i]):
            r = next(reader)
            if r not in seen_set:
                seen_set.add(r)
                never_see += 1
        cold_miss_list[i] = never_see / (break_points[i+1] - break_points[i])

    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
    draw2d(cold_miss_list, figname=figname, xticks=xticks, xlabel='time({})'.format(mode),
           ylabel='cold miss ratio(interval={})'.format(time_interval),
           title='cold miss ratio 2D plot')
    reader.reset()


def draw2d(l, **kwargs):
    """
    given a list l, plot it
    :param l:
    :param kwargs:
    :return:
    """
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
    if 'logX' in kwargs:
        plt.gca().set_xscale("log")
    if 'logY' in kwargs:
        plt.gca().set_yscale("log")

    plt.xlim((0, len(l)+1))

    plt.savefig(filename, dpi=600)
    try:
        plt.show()
    except:
        pass
    INFO("plot is saved at the same directory")
    plt.clf()


def nameMapping_2d(reader, partial_ratio=0.1, figname=None):
    """
    rename all the IDs for items in the trace for visualization of trace
    :param reader:
    :param partial_ratio: take fitst partial_ratio of trace for zooming in
    :param figname:
    :return:
    """
    # initialization
    SCATTER_POINT_LIMIT = 6000
    mapping_counter = 0
    num_of_requests = reader.get_num_total_req()
    num_of_partial = int(num_of_requests * partial_ratio)
    name_mapping = {}

    list_overall = []
    list_partial = []

    # the two ratio below is used for sampling
    adjust_ratio_overall = max(num_of_requests // SCATTER_POINT_LIMIT, 1)
    adjust_ratio_partial = max(num_of_requests * partial_ratio // SCATTER_POINT_LIMIT, 1)

    # name mapping
    for n, e in enumerate(reader):
        if e not in name_mapping:
            name_mapping[e] = mapping_counter
            mapping_counter += 1
        if n % adjust_ratio_overall == 0:
            list_overall.append(name_mapping[e])
        if n < num_of_partial and n % adjust_ratio_partial == 0:
            list_partial.append(name_mapping[e])

    # plotting
    plt.scatter(np.linspace(0, 100, len(list_overall)), list_overall)
    plt.title("mapped block versus time(overall)")
    plt.ylabel("mapped LBA")
    plt.xlabel("virtual time/%")
    if figname is None:
        new_figname = os.path.basename(reader.fileloc) + '_overall.png'
    else:
        pos = figname.rfind('.')
        new_figname = figname[:pos] + '_overall' + figname[pos:]
    plt.savefig(new_figname)

    plt.clf()
    plt.scatter(np.linspace(0, 100, len(list_partial)), list_partial)
    plt.title("renamed block versus time(part)")
    plt.ylabel("renamed block number")
    plt.xlabel("virtual time/%")
    if figname is None:
        new_figname = os.path.basename(reader.fileloc) + '_partial.png'
    else:
        pos = figname.rfind('.')
        new_figname = figname[:pos] + '_partial' + figname[pos:]
    plt.savefig(new_figname)
    plt.clf()
    INFO("plot is saved at the same directory")
    reader.reset()
