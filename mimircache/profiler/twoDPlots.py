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
from mimircache.profiler.LRUProfiler import LRUProfiler
import os
import numpy as np
from collections import defaultdict


def request_rate_2d(reader, mode, time_interval, figname="request_num.png"):
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


def popularity_2d(reader, logX=True, logY=False, cdf=True, plot_type="obj", figname=None):
    """
    plot the popularity curve of the obj in the trace
    :param reader:
    :param logX:
    :param logY:
    :param cdf:
    :param figname:
    :return:
    """
    if figname is None:
        figname = "popularity_{}.png".format(os.path.basename(reader.file_loc))

    req_freq_dict = reader.get_req_freq_distribution()
    freq_count_dict = defaultdict(int)
    max_freq = -1
    for _, v in req_freq_dict.items():
        freq_count_dict[v] += 1
        if v > max_freq:
            max_freq = v

    l = [0] * max_freq
    xlabel = "Obj Frequency"
    if plot_type.lower() == "obj":
        print(freq_count_dict)
        if not cdf:
            ylabel = "Num of Obj"
            for k, v in freq_count_dict.items():
                l[k-1] = v
        else:
            ylabel = "Num of Obj Percentage (CDF)"
            for freq, freq_count in freq_count_dict.items():
                l[freq-1] = freq_count
            for i in range(1, len(l)):
                l[i] = l[i-1]+l[i]
            for i in range(0, len(l)):
                l[i] = l[i] / l[-1]
            print(l)
    elif plot_type.lower() == "req":
        if not cdf:
            ylabel = "Num of Req"
            for freq, freq_count in freq_count_dict.items():
                l[freq-1] = freq_count * freq
        else:
            ylabel = "Num of Req Percentage (CDF)"
            for freq, freq_count in freq_count_dict.items():
                l[freq -1] = freq * freq_count
            for i in range(1, len(l)):
                l[i] = l[i-1]+l[i]
            for i in range(0, len(l)):
                l[i] = l[i] / l[-1]
    else:
        ERROR("unknown plot type {}".format(plot_type))
        return

    draw2d(l, xlabel=xlabel, ylabel=ylabel, logX=logX, logY=logY, figname=figname)
           # , yticks=ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x *100 / len(l))))
    reader.reset()



def rd_freq_popularity_2d(reader, logX=True, logY=True, cdf=False, figname="rdFreq_popularity_2d.png"):
    """
    plot the reuse distance distribution in two dimensional figure
    :param reader:
    :param logX:
    :param logY:
    :param cdf:
    :param figname:
    :return:
    """
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
    xlabel = "reuse distance frequency"
    if not cdf:
        ylabel = "num of requests"
        for k, v in rd_count_dict.items():
            l[k-1] = v
    else:
        ylabel = "num of requests (CDF)"
        for k, v in rd_count_dict.items():
            # l[-k] = v                # this is not necessary
            l[k-1] = v
        for i in range(1, len(l)):
            l[i] = l[i-1]+l[i]
        for i in range(0, len(l)):
            l[i] = l[i] / l[-1]

    draw2d(l, xlabel=xlabel, ylabel=ylabel, logX=logX, logY=logY, figname=figname,
           xticks=ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / len(l))))
    reader.reset()


def rd_popularity_2d(reader, logX=True, logY=False, cdf=True, figname="rd_popularity_2d.png"):
    """
    plot the reuse distance distribution in two dimensional figure
    :param reader:
    :param logX:
    :param logY:
    :param cdf:
    :param figname:
    :return:
    """
    rd_list = LRUProfiler(reader).get_reuse_distance()
    rd_dict = defaultdict(int)      # rd -> count
    for rd in rd_list:
        rd_dict[rd] += 1

    max_rd = -1
    for rd, _ in rd_dict.items():
        if rd > max_rd:
            max_rd = rd

    if not logX or logY or not cdf:
        WARNING("recommend using logX without logY with cdf")

    l = [0] * (max_rd + 2)
    # l = [0] * (int(math.log(max_rd, logX_base)) + 1)
    xlabel = "reuse distance"
    if not cdf:
        ylabel = "num of requests"
        for rd, rd_count in rd_dict.items():
            if rd != -1:                         # ignore cold miss
                # l[int(math.log(rd+1, logX_base))] += rd_count   # rd 0 -> pos 0, rd 1 -> pos 1
                l[rd + 1] = rd_count
    else:
        ylabel = "num of requests (CDF)"
        for rd, rd_count in rd_dict.items():
            if rd != -1:
                # l[int(math.log(rd+1, logX_base))] = v
                l[rd + 1] = rd_count
        for i in range(1, len(l)):
            l[i] = l[i-1]+l[i]
        for i in range(0, len(l)):
            l[i] = l[i] / l[-1]

    draw2d(l, xlabel=xlabel, ylabel=ylabel, logX=logX, logY=logY, figname=figname)
#            xticks=ticker.FuncFormatter(lambda x, _: '{}'.format( int(logX_base ** x - 1))))
    reader.reset()


def rt_popularity_2d(reader, granularity=10, logX=True, logX_base=1.08, logY=False, cdf=True, figname="rt_popularity_2d.png"):
    """
        plot the reuse time distribution in two dimensional figure

    :param reader:
    :param granularity:
    :param logX:
    :param logY:
    :param cdf:
    :param figname:
    :return:
    """

    last_access_time_dic = {}
    rt_dic = defaultdict(int)           # rt -> count
    line = reader.read_time_request()
    while line:
        time, req = line
        time_with_granularity = int(time/granularity)
        if req in last_access_time_dic:
            rt_dic[time_with_granularity - last_access_time_dic[req]] += 1
        last_access_time_dic[req] = time_with_granularity
        line = reader.read_time_request()

    max_rt = -1
    for rt, _ in rt_dic.items():
        if rt > max_rt:
            max_rt = rt

    if not logX or logY or not cdf:
        WARNING("recommend using logX without logY with cdf")

    l = [0] * (max_rt + 1)

    xlabel = "reuse time (unit: {})".format(granularity)
    if not cdf:
        ylabel = "num of requests"
        for rt, rt_count in rt_dic.items():
            l[rt] = rt_count
    else:
        ylabel = "num of requests (CDF)"
        for rt, rt_count in rt_dic.items():
            if rt != -1:
                l[rt] = rt_count
        for i in range(1, len(l)):
            l[i] = l[i-1] + l[i]
        for i in range(0, len(l)):
            l[i] = l[i] / l[-1]

    draw2d(l, xlabel=xlabel, ylabel=ylabel, logX=logX, logY=logY,
           # rotationXAxisLabel=True,
           figname=figname)
    reader.reset()


def hit_ratio_over_time_2d(reader, cache_size, decay_coef=0.2, time_mode="v", time_interval=10000, figname=None):
    if figname is None:
        figname = "hr_time_{}_{}_{}.png".format(os.path.basename(reader.file_loc), cache_size, decay_coef)


    p = LRUProfiler(reader)
    rd_list = p.use_precomputedRD()

    hit_ratio_list = []
    hit_ratio_overall = 0
    hit_count_current_interval = 0

    if time_mode == "v":
        for n, rd in enumerate(rd_list):
            if rd > cache_size or rd == -1:
                pass
            else:
                hit_count_current_interval += 1
            if n % time_interval == 0:
                hit_ratio_current_interval = hit_count_current_interval / time_interval
                hit_ratio_overall = hit_ratio_overall * decay_coef + hit_ratio_current_interval * (1 - decay_coef)
                hit_count_current_interval = 0
                hit_ratio_list.append(hit_ratio_overall)

    elif time_mode == "r":
        ind = 0
        req_count_current_interval = 0
        line = reader.read_time_request()
        t, req = line
        last_time_interval_cutoff = line[0]

        while line:
            last_time = t
            t, req = line
            if t - last_time_interval_cutoff > time_interval:
                hit_ratio_current_interval = hit_count_current_interval / req_count_current_interval
                hit_ratio_overall = hit_ratio_overall * decay_coef + hit_ratio_current_interval * (1 - decay_coef)
                hit_count_current_interval = 0
                req_count_current_interval = 0
                last_time_interval_cutoff = last_time
                hit_ratio_list.append(hit_ratio_overall)

            rd = rd_list[ind]
            req_count_current_interval += 1
            if rd != -1 and rd <= cache_size:
                hit_count_current_interval += 1

            line = reader.read_time_request()
            ind += 1

    draw2d(hit_ratio_list, xlabel="{} time".format({"r": "real", "v": "virtual"}.get(time_mode, "")),
           ylabel="hit ratio (decay {})".format(decay_coef),
           logX=False, logY=False, figname=figname,
           xticks=ticker.FuncFormatter(lambda x, _: '{:.0%}'.format((x ) / len(hit_ratio_list))))




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

    # plt.xlim((0, len(l)))

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    if 'xticks' in kwargs:
        plt.gca().xaxis.set_major_formatter(kwargs['xticks'])
    if 'yticks' in kwargs:
        plt.gca().yaxis.set_major_formatter(kwargs['yticks'])
    if 'xlimit' in kwargs:
        print(plt.xlim())
        plt.xlim(kwargs["xlimit"])
        print(plt.xlim())
    if 'ylimit' in kwargs:
        plt.ylim(kwargs["ylimit"])
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    # if 'logX' in kwargs and kwargs["logX"]:
    #     plt.gca().set_xscale("log")
    # if 'logY' in kwargs and kwargs["logY"]:
    #     plt.gca().set_yscale("log")
    if "rotationXAxisLabel" in kwargs:
        print("rotate")
        plt.xticks(rotation="vertical")
    plt.tight_layout()

    plt.savefig(filename, dpi=600)
    try:
        plt.show()
    except:
        pass
    if 'print_info' in kwargs and not kwargs["print_info"]:
        pass
    else:
        INFO("plot is saved at the same directory")
    plt.clf()