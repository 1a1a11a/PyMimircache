# coding=utf-8
"""
this module provides functions for all the two dimensional figure plotting, currently including:
    ### time related ###
    request_rate_2d,
    cold_miss_count_2d,
    cold_miss_ratio_2d,
    namemapping_2d (mapping block to a LBA) for visulization of scan and so on
    interval_hit_ratio_2d

    ### static ###
    popularity_2d  (frequency popularity)
    rd_freq_popularity_2d,
    rd_popularity_2d,
    rt_popularity_2d,


In time related plots, the x-axis should be real or virtual time.

    TODO:
        add percentage to rd_popularity_2d

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/10

"""


import os
import sys
import numpy as np
from collections import defaultdict
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt

from PyMimircache.const import ALLOW_C_MIMIRCACHE, INSTALL_PHASE
if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    from PyMimircache.profiler.cHeatmap import CHeatmap as Heatmap
    from PyMimircache.profiler.cLRUProfiler import CLRUProfiler as LRUProfiler
else:
    print("not all plots in twoDPlots support Py mode", file=sys.stderr)
    # raise RuntimeError("Py mode is not ready in twoDPlots")

from PyMimircache.utils.printing import *
from PyMimircache.profiler.profilerUtils import draw2d

__all__=[
    "request_rate_2d",
    "cold_miss_count_2d",
    "cold_miss_ratio_2d",
    # "namemapping_2d",
    "scan_vis_2d",
    "popularity_2d",
    "rd_freq_popularity_2d",
    "rd_popularity_2d",
    "rt_popularity_2d",
    "interval_hit_ratio_2d"
]

def request_rate_2d(reader, time_mode, time_interval,
                    figname="request_rate.png", **kwargs):
    """
    plot the number of requests per time_interval vs time
    :param reader:
    :param time_mode: either 'r' or 'v' for real time(wall-clock time) or virtual time(reference time)
    :param time_interval:
    :param figname:
    :return: the list of data points
    """

    kwargs_plot = {}
    kwargs_plot.update(kwargs)

    kwargs_plot["xlabel"] = kwargs_plot.get("xlabel", '{} Time'.format("Real" if time_mode == "r" else "Virtual"))
    kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", 'Request Rate (interval={})'.format(time_interval))
    kwargs_plot["title"] = kwargs_plot.get("title", 'Request Rate Plot')
    kwargs_plot["xticks"] = kwargs_plot.get("xticks",
                        ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points))))

    assert time_mode == 'r' or time_mode == 'v', "currently only support time_mode r and v, what time_mode are you using?"
    break_points = Heatmap.get_breakpoints(reader, time_mode, time_interval)

    l = []
    for i in range(1, len(break_points)):
        l.append(break_points[i] - break_points[i - 1])

    draw2d(l, figname=figname, **kwargs_plot)
    return l


def cold_miss_count_2d(reader, time_mode, time_interval,
                       figname="cold_miss_count2d.png", **kwargs):
    """
    plot the number of cold miss per time_interval
    :param reader:
    :param time_mode: either 'r' or 'v' for real time(wall-clock time) or virtual time(reference time)
    :param time_interval:
    :param figname:
    :return: the list of data points
    """

    kwargs_plot = {}
    kwargs_plot.update(kwargs)

    kwargs_plot["xlabel"] = kwargs_plot.get("xlabel", '{} Time'.format("Real" if time_mode == "r" else "Virtual"))
    kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", 'Cold Miss Count (interval={})'.format(time_interval))
    kwargs_plot["title"] = kwargs_plot.get("title", 'Cold Miss Count 2D plot')
    kwargs_plot["xticks"] = kwargs_plot.get("xticks",
                        ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points))))
    kwargs_plot["label"] = kwargs_plot.get("label", "Cold Miss Count")

    assert time_mode == 'r' or time_mode == 'v', "currently only support time_mode r and v, what time_mode are you using?"
    break_points = Heatmap.get_breakpoints(reader, time_mode, time_interval)

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

    draw2d(cold_miss_list, figname=figname, **kwargs_plot)
    reader.reset()
    return cold_miss_list


def cold_miss_ratio_2d(reader, time_mode, time_interval,
                       figname="cold_miss_ratio2d.png", **kwargs):
    """
    plot the percent of cold miss per time_interval
    :param reader:
    :param time_mode: either 'r' or 'v' for real time(wall-clock time) or virtual time(reference time)
    :param time_interval:
    :param figname:
    :return: the list of data points
    """

    kwargs_plot = {}
    kwargs_plot.update(kwargs)

    kwargs_plot["xlabel"] = kwargs_plot.get("xlabel", '{} Time'.format("Real" if time_mode == "r" else "Virtual"))
    kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", 'Cold Miss Ratio (interval={})'.format(time_interval))
    kwargs_plot["title"] = kwargs_plot.get("title", 'Cold Miss Ratio 2D plot')
    kwargs_plot["xticks"] = kwargs_plot.get("xticks",
                        ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points))))

    assert time_mode == 'r' or time_mode == 'v', \
        "currently only support time_mode r and v, unknown time_mode {}".format(time_mode)
    break_points = Heatmap.get_breakpoints(reader, time_mode, time_interval)

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

    draw2d(cold_miss_list, figname=figname, **kwargs_plot)
    reader.reset()
    return cold_miss_list


def scan_vis_2d(reader, partial_ratio=0.1, figname=None, **kwargs):
    """
    rename all the ojbID for items in the trace for visualization of trace
    so the first obj is renamed to 1, the second obj is renamed to 2, etc.
    Notice that it is not first request, second request...

    :param reader:
    :param partial_ratio: take fitst partial_ratio of trace for zooming in
    :param figname:
    :return:
    """

    # initialization
    SCATTER_POINT_LIMIT = 6000
    mapping_counter = 0
    num_of_requests = reader.get_num_of_req()
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
    if 'pointSize' in kwargs.keys():
        point_size = kwargs.get('pointSize', False)

        if isinstance(point_size, bool):
            WARNING("Undefined or unknown pointSize '{}', using default value".format(point_size))
            point_size = plt.rcParams['lines.markersize'] ** 2
        elif not isinstance(point_size, (int, float)):
            WARNING("Undefined or unknown pointSize '{}', using default value".format(point_size))
            point_size = plt.rcParams['lines.markersize'] ** 2
        else:
            pass
    else:
        point_size = plt.rcParams['lines.markersize'] ** 2

    plt.scatter(np.linspace(0, 100, len(list_overall)), list_overall, s=point_size)
    plt.title("mapped block versus time(overall)")
    plt.ylabel("mapped LBA")
    plt.xlabel("virtual time/%")

    if figname is None:
        base_figname = os.path.basename(reader.fileloc)
    else:
        pos = figname.rfind('.')
        base_figname = figname[:pos] + '_overall' + figname[pos:]
    plt.tight_layout()
    plt.savefig("{}_overall.png".format(base_figname))

    plt.clf()
    plt.scatter(np.linspace(0, 100, len(list_partial)), list_partial, s=point_size)
    plt.title("renamed block versus time(part)")
    plt.ylabel("renamed block number")
    plt.xlabel("virtual time/%")
    plt.tight_layout()
    plt.savefig("{}_partial.png".format(base_figname))
    plt.clf()
    INFO("mapping plot is saved")
    reader.reset()


def popularity_2d(reader, logX=True, logY=False, cdf=True, plot_type="obj",
                  figname="popularity_2d.png", **kwargs):
    """
    plot the popularity curve of the obj in the trace
    X axis is object frequency,
    Y axis is either obj percentage or request percentage depending on plot_type

    :param reader:
    :param logX:
    :param logY:
    :param cdf:
    :param plot_type:
    :param figname:
    :return: the list of data points
    """

    kwargs_plot = {}
    kwargs_plot.update(kwargs)

    kwargs_plot["logX"], kwargs_plot["logY"] = logX, logY
    kwargs_plot["cdf"] = cdf
    kwargs_plot["xlabel"] = kwargs_plot.get("xlabel", "Obj Frequency")


    req_freq_dict = reader.get_req_freq_distribution()
    freq_count_dict = defaultdict(int)
    max_freq = -1
    for v in req_freq_dict.values():
        freq_count_dict[v] += 1
        if v > max_freq:
            max_freq = v

    l = [0] * max_freq
    if plot_type.lower() == "obj":
        if not cdf:
            kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", "Obj Percentage")
            for k, v in freq_count_dict.items():
                l[k-1] = v
        else:
            kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", "Obj Percentage (CDF)")
            for freq, freq_count in freq_count_dict.items():
                l[freq-1] = freq_count
            for i in range(1, len(l)):
                l[i] = l[i-1]+l[i]
            for i in range(0, len(l)):
                l[i] = l[i] / l[-1]
    elif plot_type.lower() == "req":
        if not cdf:
            kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", "Request Percentage")
            for freq, freq_count in freq_count_dict.items():
                l[freq-1] = freq_count * freq
        else:
            kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", "Req Percentage (CDF)")
            for freq, freq_count in freq_count_dict.items():
                l[freq -1] = freq * freq_count
            for i in range(1, len(l)):
                l[i] = l[i-1]+l[i]
            for i in range(0, len(l)):
                l[i] = l[i] / l[-1]
    else:
        ERROR("unknown plot type {}".format(plot_type))
        return

    draw2d(l, figname=figname, **kwargs_plot)
    reader.reset()
    return l


def rd_freq_popularity_2d(reader, logX=True, logY=True, cdf=False,
                          figname="rdFreq_popularity_2d.png", **kwargs):
    """
    plot the reuse distance distribution in a two dimensional figure,
    X axis is reuse distance frequency
    Y axis is the number of requests in percentage
    :param reader:
    :param logX:
    :param logY:
    :param cdf:
    :param figname:
    :return: the list of data points
    """

    kwargs_plot = {}
    kwargs_plot.update(kwargs)

    kwargs_plot["logX"], kwargs_plot["logY"] = logX, logY
    kwargs_plot["cdf"] = cdf
    kwargs_plot["xlabel"] = kwargs_plot.get("xlabel", "Reuse Distance Frequency")
    kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", "Requests Percentage")
    kwargs_plot["xticks"] = kwargs_plot.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / len(l))))


    rd_list = LRUProfiler(reader).get_reuse_distance()
    rd_dict = defaultdict(int)      # rd -> count
    for rd in rd_list:
        rd_dict[rd] += 1

    rd_count_dict = defaultdict(int)        # rd_count -> count of rd_count
    max_freq = -1
    for _, v in rd_dict.items():
        rd_count_dict[v] += 1
        if v > max_freq:
            max_freq = v

    l = [0] * max_freq
    if not cdf:
        for k, v in rd_count_dict.items():
            l[k-1] = v
    else:
        kwargs_plot["ylabel"] = kwargs.get("ylabel", "Requests Percentage (CDF)")
        for k, v in rd_count_dict.items():
            # l[-k] = v                # this is not necessary
            l[k-1] = v
        for i in range(1, len(l)):
            l[i] = l[i-1]+l[i]
        for i in range(0, len(l)):
            l[i] = l[i] / l[-1]

    draw2d(l, figname=figname, **kwargs_plot)
    reader.reset()
    return l


def rd_popularity_2d(reader, logX=True, logY=False, cdf=True,
                     figname="rd_popularity_2d.png", **kwargs):
    """
    plot the reuse distance distribution in two dimension, cold miss is ignored
    X axis is reuse distance
    Y axis is number of requests (not in percentage)
    :param reader:
    :param logX:
    :param logY:
    :param cdf:
    :param figname:
    :return: the list of data points
    """

    if not logX or logY or not cdf:
        WARNING("recommend using logX without logY with cdf")

    kwargs_plot = {}
    kwargs_plot.update(kwargs)

    kwargs_plot["logX"], kwargs_plot["logY"] = logX, logY
    kwargs_plot["cdf"] = cdf
    kwargs_plot["xlabel"] = kwargs_plot.get("xlabel", "Reuse Distance")
    kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", "Num of Requests")

    rd_list = LRUProfiler(reader).get_reuse_distance()
    rd_dict = defaultdict(int)      # rd -> count
    for rd in rd_list:
        rd_dict[rd] += 1

    max_rd = -1
    for rd, _ in rd_dict.items():
        if rd > max_rd:
            max_rd = rd

    l = [0] * (max_rd + 2)
    if not cdf:
        for rd, rd_count in rd_dict.items():
            if rd != -1:                         # ignore cold miss
                l[rd + 1] = rd_count
    else:
        kwargs_plot["ylabel"] = kwargs.get("ylabel", "Num of Requests (CDF)")
        for rd, rd_count in rd_dict.items():
            if rd != -1:
                l[rd + 1] = rd_count
        for i in range(1, len(l)):
            l[i] = l[i-1]+l[i]
        for i in range(0, len(l)):
            l[i] = l[i] / l[-1]

    draw2d(l, figname=figname, **kwargs_plot)
    reader.reset()
    return l


def rt_popularity_2d(reader, granularity=10, logX=True, logY=False, cdf=True,
                     figname="rt_popularity_2d.png", **kwargs):
    """
    plot the reuse time distribution in the trace
    X axis is reuse time,
    Y axis is number of requests (not in percentage)

    :param reader:
    :param granularity:
    :param logX:
    :param logY:
    :param cdf:
    :param figname:
    :param kwargs: time_bin
    :return: the list of data points
    """

    kwargs_plot = {}
    kwargs_plot.update(kwargs)

    kwargs_plot["logX"], kwargs_plot["logY"] = logX, logY
    kwargs_plot["cdf"] = cdf
    kwargs_plot["xlabel"] = kwargs_plot.get("xlabel", "Reuse Time (unit: {})".format(granularity))
    kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", "Num of Requests")

    last_access_time_dic = {}
    rt_dic = defaultdict(int)           # rt -> count
    line = reader.read_time_req()
    while line:
        time, req = line
        time_with_granularity = int(time/granularity)
        if req in last_access_time_dic:
            rt_dic[time_with_granularity - last_access_time_dic[req]] += 1
        last_access_time_dic[req] = time_with_granularity
        line = reader.read_time_req()

    max_rt = -1
    for rt, _ in rt_dic.items():
        if rt > max_rt:
            max_rt = rt

    if not logX or logY or not cdf:
        WARNING("recommend using logX without logY with cdf")

    # check granularity
    if max_rt > 1000000:
        WARNING("max reuse time {} larger than 1000000, "
                "are you setting the right time granularity? "
                "Current granularity {}".format(max_rt, granularity))

    l = [0] * (max_rt + 1)

    if not cdf:
        for rt, rt_count in rt_dic.items():
            if rt != -1:
                l[rt] = rt_count
    else:
        kwargs_plot["ylabel"] = kwargs.get("ylabel", "Num of Requests (CDF)")
        for rt, rt_count in rt_dic.items():
            if rt != -1:
                l[rt] = rt_count
        print("l {}".format(len(l)))
        for i in range(1, len(l)):
            l[i] = l[i-1] + l[i]
        for i in range(0, len(l)):
            l[i] = l[i] / l[-1]

    draw2d(l, figname=figname, **kwargs_plot)
    reader.reset()
    return l


def interval_hit_ratio_2d(reader, cache_size, decay_coef=0.8,
                          time_mode="v", time_interval=10000,
                          figname="IHRC_2d.png",
                          **kwargs):
    """
    The hit ratio curve over time interval, each pixel in the plot represents the
    exponential weight moving average (ewma) of hit ratio of the interval

    :param reader:
    :param cache_size:
    :param decay_coef: used in ewma
    :param time_mode:
    :param time_interval:
    :param figname:
    :return: the list of data points
    """


    p = LRUProfiler(reader)
    # reuse distance list
    rd_list = p.get_reuse_distance()

    hit_ratio_list = []
    ewma_hit_ratio = 0
    hit_cnt_interval = 0

    if time_mode == "v":
        for n, rd in enumerate(rd_list):
            if rd > cache_size or rd == -1:
                # this is a miss
                pass
            else:
                hit_cnt_interval += 1
            if n % time_interval == 0:
                hit_ratio_interval = hit_cnt_interval / time_interval
                ewma_hit_ratio = ewma_hit_ratio * decay_coef + hit_ratio_interval * (1 - decay_coef)
                hit_cnt_interval = 0
                hit_ratio_list.append(ewma_hit_ratio)

    elif time_mode == "r":
        ind = 0
        req_cnt_interval = 0

        # read time and request label
        line = reader.read_time_req()
        t, req = line
        last_time_interval_cutoff = line[0]

        while line:
            last_time = t
            t, req = line
            if t - last_time_interval_cutoff > time_interval:
                hit_ratio_interval = hit_cnt_interval / req_cnt_interval
                ewma_hit_ratio = ewma_hit_ratio * decay_coef + hit_ratio_interval * (1 - decay_coef)
                hit_cnt_interval = 0
                req_cnt_interval = 0
                last_time_interval_cutoff = last_time
                hit_ratio_list.append(ewma_hit_ratio)

            rd = rd_list[ind]
            req_cnt_interval += 1
            if rd != -1 and rd <= cache_size:
                hit_cnt_interval += 1

            line = reader.read_time_req()
            ind += 1

    kwargs_plot = {}
    kwargs_plot.update(kwargs)

    kwargs_plot["logX"] = kwargs_plot.get("logX", False)
    kwargs_plot["logY"] = kwargs_plot.get("logY", False)
    kwargs_plot["xlabel"] = kwargs_plot.get("xlabel", "{} Time".format(
        {"r": "Real", "v": "Virtual"}.get(time_mode, "")))
    kwargs_plot["ylabel"] = kwargs_plot.get("ylabel", "Interval Hit Ratio (decay {})".format(decay_coef))

    kwargs_plot["xticks"] = kwargs_plot.get("xticks", ticker.FuncFormatter(
        # both works
        # lambda x, _: '{:.0f}%'.format(x * 100 / len(hit_ratio_list))))
        lambda x, _: '{:.0%}'.format(x / len(hit_ratio_list))))

    reader.reset()
    draw2d(hit_ratio_list, figname=figname, **kwargs_plot)

    return hit_ratio_list




def draw2d_old(l, **kwargs):
    """
    given a list l, plot it in two dimension
    :param l:
    :param kwargs:
    :return:
    """
    filename = kwargs.get("figname", "2dPlot.png")

    if "plot_type" in kwargs:
        if kwargs['plot_type'] == "scatter":
            print("scatter plot")
            plt.scatter([i+1 for i in range(len(l))], l)
    else:
        if 'logX' in kwargs and kwargs["logX"]:
            if 'logY' in kwargs and kwargs["logY"]:
                plt.loglog(l)
            else:
                plt.semilogx(l)
        else:
            if 'logY' in kwargs and kwargs["logY"]:
                plt.semilogy(l)
            else:
                plt.plot(l)

    # set label
    if kwargs.get("xlabel", None):
        plt.xlabel(kwargs['xlabel'])
    if kwargs.get('ylabel', None):
        plt.ylabel(kwargs['ylabel'])

    # set tick
    if kwargs.get('xticks', None):
        xticks = kwargs['xticks']
        if isinstance(xticks, list) or isinstance(xticks, tuple):
            plt.xticks(*xticks)
        elif callable(xticks):
            plt.gca().xaxis.set_major_formatter(xticks)
        else:
            WARNING("unknown xticks {}".format(xticks))

    if kwargs.get('yticks', None):
        yticks = kwargs['yticks']
        if isinstance(yticks, list) or isinstance(yticks, tuple):
            plt.yticks(*yticks)
        elif callable(yticks):
            plt.gca().yaxis.set_major_formatter(yticks)
        else:
            WARNING("unknown yticks {}".format(yticks))

    # set limit
    if kwargs.get("xlimit", None):
        plt.xlim(kwargs["xlimit"])
    if kwargs.get('ylimit', None):
        plt.ylim(kwargs["ylimit"])

    # set title
    if kwargs.get('title', None):
        plt.title(kwargs['title'])

    # if x axis label are too long, then rotate it
    if 'rotateXAxisTick' in kwargs.keys():
        xrotate = kwargs['rotateXAxisTick']

        if isinstance(xrotate, bool):
            plt.xticks(rotation="vertical")
        elif isinstance(xrotate, (int, float)):
            plt.xticks(rotation=xrotate)
        else:
            plt.xticks(rotation="vertical")
            WARNING("unknown rotateXAxisTick {}".format(xrotate))

    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    try: plt.show()
    except: pass
    INFO("plot is saved as {}".format(filename))
    if not kwargs.get("no_clear", False):
        plt.clf()
