# coding=utf-8

from mimircache import *
from math import ceil, log2

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import mimircache.c_eviction_stat as c_eviction_stat
from mimircache.const import *
from mimircache.utils.printing import *
from mimircache.utils.prepPlotParams import *



def eviction_stat_reuse_dist_plot(reader, algorithm, cache_size, mode, time_interval, cache_params=None, **kwargs):
    """

    :param reader:
    :param algorithm:
    :param cache_size:
    :param mode:
    :param time_interval:
    :param cache_params:
    :param kwargs:
    :return:
    """
    plot_params = prepPlotParams("reuse distance of evicted elements by {}".format(algorithm),
                                 "time({})".format(mode), "reuse dist/cache size",
                                 "eviction_stat_reuse_dist_{}_{}_{}_{}_{}_{}.png".format(
                                     reader.file_loc[reader.file_loc.rfind('/')+1:], algorithm, cache_size,
                                     mode, time_interval, cache_params), **kwargs)

    alg = cache_alg_mapping[algorithm.lower()]
    assert alg=="Optimal", "Currently only Optimal is supported"

    # get reuse distance of evicted elements by given algorithm
    rd_array = c_eviction_stat.get_stat(reader.cReader, algorithm=alg, cache_size=cache_size, stat_type="reuse_dist")

    # generate break points for bucketing the reuse_dist array
    bp = cHeatmap().getBreakpoints(reader, mode, time_interval)

    pos = 1
    count = 0
    matrix = np.zeros( (len(bp), 20) )

    for i in range(len(rd_array)):
        # no eviction at this timestamp
        if rd_array[i] < 0:
            if i == bp[pos]:
                for j in range(20):
                    if count:
                        matrix[pos-1][j] /= count
                    else:
                        matrix[pos-1][j] = np.nan
                pos += 1
                count = 0
            continue

        # check the bucket for the reuse dist of the evicted element
        y = float(rd_array[i]) / cache_size
        if y < 1:
            y = ceil(y * 10)
        elif y > 10:
            y = 19
        else:
            y = ceil(y)+9
        matrix[pos-1][y] += 1
        count += 1

        if i == bp[pos]:
            for j in range(20):
                matrix[pos - 1][j] /= count
            pos += 1
            count = 0

        # y could be zero, which means originally rd_array[i]=0, meaning MRU

    for j in range(20):
        if count!=0:
            matrix[pos - 1][j] /= count
        else:
            matrix[pos - 1][j] = np.nan

    majorLocator = MultipleLocator(2)
    minorLocator = MultipleLocator(1)


    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(bp)))
    yticks = ticker.IndexFormatter([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


    plt.gca().xaxis.set_major_formatter(xticks)
    plt.gca().yaxis.set_major_formatter(yticks)
    plt.gca().yaxis.set_major_locator(majorLocator)
    plt.gca().yaxis.set_minor_locator(minorLocator)

    img = plt.imshow(matrix.T, interpolation='nearest', origin='lower', aspect='auto', vmin=0, vmax=1)
    cb = plt.colorbar(img)
    plt.title(plot_params['title'])
    plt.xlabel(plot_params['xlabel'])
    plt.ylabel(plot_params['ylabel'])
    plt.savefig(plot_params['figname'], dpi=600)
    try:
        plt.show()
    except:
        pass
    plt.clf()
    INFO("plot is saved at the same directory")


def eviction_stat_freq_plot(reader, algorithm, cache_size, mode, time_interval,
                       accumulative=True, cache_params=None, yscale=6, **kwargs):

    plot_params = prepPlotParams("frequency of evicted elements by {}".format(algorithm),
                                 "time({})".format(mode), "frequency",
                                 "eviction_stat_freq_{}_{}_{}_{}_{}_{}_{}.png".format(
                                     reader.file_loc[reader.file_loc.rfind('/')+1:], algorithm, cache_size,
                                     mode, time_interval, accumulative, cache_params), **kwargs)

    alg = cache_alg_mapping[algorithm.lower()]
    stat_type = "freq"
    if accumulative:
        stat_type = "accumulative_freq"
    assert alg=="Optimal", "Currently only Optimal is supported"
    freq_array = c_eviction_stat.get_stat(reader.cReader, algorithm=alg, cache_size=cache_size, stat_type=stat_type)
    bp = cHeatmap().getBreakpoints(reader, mode, time_interval)

    pos = 1
    count = 0
    matrix = np.zeros( (len(bp), yscale) )

    for i in range(len(freq_array)):
        # no eviction at this timestamp
        if freq_array[i] < 0:
            if i == bp[pos]:
                for j in range(yscale):
                    if count:
                        matrix[pos-1][j] /= count
                    else:
                        matrix[pos-1][j] = np.nan
                pos += 1
                count = 0
            # else nothing needs to be done, just continue
            continue

        # check the bucket for the reuse dist of the evicted element
        y = ceil(log2(freq_array[i]))       #freq_array[i] always >= 1
        if y > yscale-1:
            y = yscale-1
        matrix[pos-1][y] += 1
        count += 1

        if i == bp[pos]:
            for j in range(yscale):
                if count:
                    matrix[pos - 1][j] /= count
                else:
                    matrix[pos - 1][j] = np.nan
            pos += 1
            count = 0


    for j in range(yscale):
        if count:
            matrix[pos - 1][j] /= count
        else:
            matrix[pos - 1][j] = np.nan

    majorLocator = MultipleLocator(2)
    minorLocator = MultipleLocator(1)

    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(bp)))
    yticks = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(2 ** x))


    plt.gca().xaxis.set_major_formatter(xticks)
    plt.gca().yaxis.set_major_formatter(yticks)
    plt.gca().yaxis.set_major_locator(majorLocator)
    plt.gca().yaxis.set_minor_locator(minorLocator)

    img = plt.imshow(matrix.T, interpolation='nearest', origin='lower', aspect='auto', vmin=0, vmax=1)
    cb = plt.colorbar(img)
    plt.title(plot_params['title'])
    plt.xlabel(plot_params['xlabel'])
    plt.ylabel(plot_params['ylabel'])
    plt.savefig(plot_params['figname'], dpi=600)
    try:
        plt.show()
    except:
        pass
    plt.clf()
    INFO("plot is saved at the same directory")
