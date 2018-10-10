# coding=utf-8
"""
    This module provides some common utils shared by profilers


    Author: Jason Yang <peter.waynechina@gmail.com> 2017/10

"""



import os
import math
import time
import pickle
import numpy as np
import matplotlib
# matplotlib.use("Agg")
# pypy3 fails on this

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PyMimircache.utils.printing import *


def get_breakpoints(reader, time_mode, time_interval=-1, num_of_pixel_of_time_dim=-1, **kwargs):

    """
    retrieve the breakpoints given time_mode and time_interval or num_of_pixel_of_time_dim,
    break point breaks the trace into chunks of given time_interval

    :param reader: reader for reading trace
    :param time_mode: either real time (r) or virtual time (v)
    :param time_interval: the intended time_interval of data chunk
    :param num_of_pixel_of_time_dim: the number of chunks, this is used when it is hard to estimate time_interval,
                            you only need specify one, either num_of_pixel_of_time_dim or time_interval
    :param kwargs: not used now
    :return: a numpy list of break points begin with 0, ends with total_num_requests
    """

    assert time_interval!=-1 or num_of_pixel_of_time_dim!=-1, \
        "please specify at least one of the following: time_interval, num_of_pixel_of_time_dim"

    bp = []
    if time_mode == "v":
        num_req = reader.get_num_of_req()
        if time_interval == -1:
            time_interval = int(math.ceil(num_req / num_of_pixel_of_time_dim) + 1)

        for i in range(int(math.ceil(num_req/time_interval))):
            bp.append(i * time_interval)
        if bp[-1] != num_req:
            bp.append(num_req)

    elif time_mode == "r":
        time_column = getattr(reader, "time_column", -1)
        assert time_column!=-1, "provided reader does not have time column"

        if time_interval == -1:
            first_ts = reader.read_first_req()[time_column-1]
            last_ts = reader.read_last_req()[time_column-1]
            time_interval = int(math.ceil((last_ts - first_ts) / num_of_pixel_of_time_dim + 1))

        bp.append(0)
        ind = 0
        line = reader.read_time_req()
        last_ts = line[0]
        while line:
            if line[0] - last_ts > time_interval:
                bp.append(ind)
                last_ts = line[0]
            line = reader.read_time_req()
            ind += 1
        if bp[-1] != ind:
            bp.append(ind)
    else:
        raise RuntimeError("unknown time_mode {}".format(time_mode))

    return bp


def util_plotHRC(x_list, hit_ratio, **kwargs):
    """
    plot hit ratio curve of the given trace under given algorithm
    :param kwargs: figname, cache_unit_size (unit: Byte), no_clear, no_save
    :return:
    """

    kwargs["figname"] = kwargs.get("figname", "HRC.png")
    kwargs["xlabel"] = kwargs.get("xlabel", "Cache Size (Items)")
    kwargs["ylabel"] = kwargs.get("ylabel", "Hit Ratio")
    kwargs["title"]  = kwargs.get("title", "Hit Ratio Curve")

    cache_unit_size = kwargs.get("cache_unit_size", 0)
    if cache_unit_size != 0:
        kwargs["xlabel"] = "Cache Size (MB)"
        ffm = ticker.FuncFormatter(lambda x, p: int(x * cache_unit_size // 1024 // 1024))
        plt.gca().xaxis.set_major_formatter(ffm)

    draw2d(x_list, hit_ratio, **kwargs)
    return



def draw2d(*args, **kwargs):


    figname = kwargs.get("figname", "2dPlot.png")

    if "plot_type" in kwargs:
        if kwargs['plot_type'] == "scatter":
            l = args[0]
            plt.scatter([i+1 for i in range(len(l))], l, label=kwargs.get("label", None))
    else:
        if 'logX' in kwargs and kwargs["logX"]:
            if 'logY' in kwargs and kwargs["logY"]:
                plt.loglog(*args, label=kwargs.get("label", None))
            else:
                plt.semilogx(*args, label=kwargs.get("label", None))
        else:
            if 'logY' in kwargs and kwargs["logY"]:
                plt.semilogy(*args, label=kwargs.get("label", None))
            else:
                plt.plot(*args, label=kwargs.get("label", None))

    set_fig(**kwargs)

    if not kwargs.get("no_save", False):
        # if folder does not exist, create the folder
        dname = os.path.dirname(figname)
        if dname and not os.path.exists(dname):
            os.makedirs(dname)
        plt.savefig(figname, dpi=600)
        if not kwargs.get("no_print_info", False):
            INFO("plot is saved as {}".format(figname))

    if not kwargs.get("no_show", False):
        try: plt.show()
        except: pass

    if not kwargs.get("no_clear", False):
        plt.clf()


def draw_heatmap(plot_array, **kwargs):
    filename = kwargs.get("figname", 'heatmap.png')

    try:
        imshow_kwargs = kwargs.get("imshow_kwargs", {})
        if "cmap" not in imshow_kwargs:
            imshow_kwargs["cmap"] = plt.cm.jet
        else:
            imshow_kwargs["cmap"] = plt.get_cmap(imshow_kwargs["cmap"])
        imshow_kwargs["cmap"].set_bad(color='white', alpha=1.)

        img = plt.imshow(plot_array, interpolation='nearest', origin='lower',
                         aspect='auto', **imshow_kwargs)

        cb = plt.colorbar(img)
        set_fig(no_legend=True, **kwargs)

        if not kwargs.get("no_save", False):
            plt.savefig(filename, dpi=600)
            INFO("plot is saved as {}".format(filename))
        if not kwargs.get("no_show", False):
            try: plt.show()
            except: pass
        if not kwargs.get("no_clear", False):
            try: plt.clf()
            except: pass

    except Exception as e:
        try:
            t = int(time.time())
            with open("/tmp/heatmap.{}.pickle".format(t), 'wb') as ofile:
                pickle.dump(plot_array, ofile)
            WARNING("plotting using imshow failed: {}, "
                    "now try to save the plotting data to /tmp/heatmap.{}.pickle".format(e, t))
        except Exception as e:
            ERROR("failed to save plotting data")

        try:
            cmap = plt.get_cmap("Oranges")
            plt.pcolormesh(plot_array.T, cmap=cmap)
            plt.savefig(filename)
        except Exception as e:
            WARNING("further plotting using pcolormesh failed" + str(e))


def set_fig(**kwargs):
    """
    change figures
    :param kwargs:
    :return:
    """

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

    # legend
    if not kwargs.get("no_legend", False):
        plt.legend(loc="best")

    # tight layout
    if kwargs.get("tight_layout", True):
        plt.tight_layout()
