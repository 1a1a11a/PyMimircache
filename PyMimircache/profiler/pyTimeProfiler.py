# coding=utf-8
"""

    This module is currently used internally to plot multiple IHRC on the same fig,
    the multiple IHRC are usually related, for exmaple,
    IHRC of the same trace with different parameters (algorithms, cache size, cache replacement algorithm parameters)

    This module is named timeProfiler because it should be able to dealing with streaming data in the future,
    and currenly it keeps plotting as the trace is read in, it won't wait until all data ready to plot


    Author: Jason <peter.waynechina@gmail.com>


"""


import time
import json
import pickle
from queue import Empty
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor, as_completed

import PyMimircache
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from PyMimircache.profiler.profilerUtils import get_breakpoints
from PyMimircache.profiler.profilerUtils import draw2d
from PyMimircache.const import cache_name_to_class
from copy import copy


def _cal_interval_hit_count_subprocess(
        time_mode,
        time_interval,
        ihc_queue,
        cache_class,
                              cache_size,
                              reader_class,
                              reader_params,
                              cache_params=None):
    """
    subprocess for simulating a cache, this will be used as init func for simulating a cache,
    it reads data from reader and calculates the number of hits and misses

    :param cache_class: the __class__ attribute of cache, this will be used to create cache instance
    :param cache_size:  size of cache
    :param reader_class: the __class__ attribute of reader, this will be used to create local reader instance
    :param reader_params:   parameters for reader, used in creating local reader instance
    :param cache_params:    parameters for cache, used in creating cache
    :return: a tuple of number of hits and number of misses
    """

    if cache_params is None:
        cache_params = {}
    process_reader = reader_class(**reader_params)
    cache_params["cache_size"] = cache_params.get("cache_size", cache_size)
    cache = cache_class(**cache_params)
    n_hits = 0
    n_misses = 0

    if time_mode == "v":
        for n, req in enumerate(process_reader):
            hit = cache.access(req, )
            if hit:
                n_hits += 1
            else:
                n_misses += 1
            if n !=0 and n % time_interval == 0:
                ihc_queue.put((n_hits, n_misses))
                n_hits = 0
                n_misses = 0

    elif time_mode == "r":
        line = process_reader.read_time_req()
        last_ts = line[0]
        while line:
            t, req = line
            hit = cache.access(req, )
            if hit:
                n_hits += 1
            else:
                n_misses += 1
            while t - last_ts > time_interval:
                ihc_queue.put((n_hits, n_misses))
                n_hits = 0
                n_misses = 0
                last_ts = last_ts + time_interval
            line = process_reader.read_time_req()

    else:
        raise RuntimeError("unknown time_mode {}".format(time_mode))


    process_reader.close()
    ihc_queue.close()
    return True


def convert_to_name(cache_class, param=None, **kwargs):
    name = str(cache_class)
    name = name[name.rfind(".")+1 : name.rfind("'")]
    if param:
        name += ", "
        if isinstance(param, dict):
            name += "_".join(["{}".format(v) for k, v in param.items() if (isinstance(v, list) and len(v) < 8) or len(str(v)) < 16])
            # name += ", ".join(["{}: {}".format(k, v) for k, v in param.items() if (not isinstance(v, list)) and len(str(v)) < 8])
            # name += ", ".join(["{}: {}".format(k, v) for k, v in param.items() if (not isinstance(v, list)) or len(v) < 5])
        elif isinstance(param, str):
            name += param
    return name


def _plot_ihrc(cache_classes, param_list, hr_dict, time_mode, figname, last_one=False):
    """
        This function does the plotting after computation,

    :param param_list:
    :param hr_dict:
    :param time_mode:
    :param figname:
    :param last_one:
    :return:
    """


    length = len(hr_dict[list(hr_dict.keys())[0]])
    if length == 0:
        return


    tick = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / length))
    # plot_kwargs = {"xlabel": "{} Time".format("Real" if time_mode == "r" else "Virtual"),
    #                "ylabel": "Hit Ratio", "no_clear": True, "figname": figname}
    #
    # if last_one:
    #     # whether this is the last plot since the plotting happens every time_interval
    #     plot_kwargs["xticks"] = tick
    #
    # for i in range(len(param_list)):
    #     param = convert_to_name(param_list[i])
    #     draw2d(hr_dict[param], label=param, **plot_kwargs)
    # plt.clf()


    if last_one:
        # whether this is the last plot since the plotting happens every time_interval
        for i in range(len(param_list) - 1):
            param = convert_to_name(cache_classes[i], param_list[i])
            draw2d(hr_dict[param], label=param, xlabel="{} Time".format("Real" if time_mode == "r" else "Virtual"),
                   ylabel="Hit Ratio", xticks=tick, no_clear=True, no_save=True)
        param = convert_to_name(cache_classes[-1], param_list[-1])
        draw2d(hr_dict[param], label=param, xlabel="{} Time".format("Real" if time_mode == "r" else "Virtual"),
                   ylabel="Hit Ratio", xticks=tick, figname=figname)

    else:
        for i in range(len(param_list) - 1):
            param = convert_to_name(cache_classes[i], param_list[i])
            draw2d(hr_dict[param], label=param, xlabel="{} Time".format("Real" if time_mode == "r" else "Virtual"),
                   ylabel="Hit Ratio", no_clear=True, no_save=True)
        param = convert_to_name(cache_classes[-1], param_list[-1])
        draw2d(hr_dict[param], label=param, xlabel="{} Time".format("Real" if time_mode == "r" else "Virtual"),
                   ylabel="Hit Ratio", figname=figname, no_print_info=False)




def plot_IHR(reader,
             time_mode,
             time_interval,
             compare_cache_sizes=(),
             compare_algs=(),
             compare_cache_params=(),
             cache_size=-1, alg=None, cache_param=None, figname="IHRC.png", **kwargs):
    """
    This is the function that does the plotting, it plots interval hit ratio curve of
    different parameters of the same trace on the same plot, and it plots every five seconds

    currrent supported comparisons: cache size, different algorithms, different cache params

    ::NOTE: there is no support for combination of different comparisons, for example,
    you cannot compare cache size and cache replacement algorithm at the same time

    :param reader:
    :param time_mode:
    :param time_interval:
    :param compare_cache_sizes:
    :param compare_algs:
    :param compare_cache_params:
    :param cache_size:
    :param alg:
    :param cache_param:
    :param figname:
    :param kwargs:
    :return:
    """

    assert len(compare_cache_sizes) * len(compare_algs) * len(compare_cache_params) == 0, \
                "You can only specify either compare_cache_sizes or compare_algs or compare_cache_params"

    reader_params = reader.get_params()
    reader_params["open_c_reader"] = False


    cache_classes = []
    param_list = []
    if len(compare_cache_sizes):
        assert alg is not None, "please provide alg for profiling"
        cache_class = alg
        if isinstance(alg, str):
            cache_class = cache_name_to_class(alg)
        cache_classes = [cache_class] * len(compare_cache_sizes)
        param_list = []
        if cache_param is None:
            cache_param = {}
        for cache_size in compare_cache_sizes:
            cache_param["cache_size"] = cache_size
            param_list.append(copy(cache_param))


    elif len(compare_cache_params):
        assert cache_size != -1, "Please provide cache size for profiling"
        assert alg is not None, "Please provide cache algorithm for profiling"

        cache_class = alg
        if isinstance(alg, str):
            cache_class = cache_name_to_class(alg)
        cache_classes = [cache_class] * len(compare_cache_params)
        param_list = compare_cache_params



    elif len(compare_algs):
        for i, cache_class in enumerate(compare_algs):
            if isinstance(cache_class, str):
                cache_class = cache_name_to_class(cache_class)
            current_cache_param = None
            if cache_param is not None and len(cache_param) > i:
                current_cache_param = cache_param[i]

            cache_classes.append(cache_class)
            param_list.append(current_cache_param)

        assert cache_size != -1, "Please provide cache size for profiling"

    else:
        raise RuntimeError("unknown compare items")



    queue_dict = {convert_to_name(cache_classes[i], param_list[i]): Queue() for i in range(len(cache_classes))}
    hr_dict = {convert_to_name(cache_classes[i], param_list[i]): [] for i in range(len(cache_classes))}
    processes = {}
    finished = 0

    for i in range(len(cache_classes)):
        cache_class = cache_classes[i]
        current_cache_param = param_list[i]

        p = Process(target=_cal_interval_hit_count_subprocess,
                    args=(time_mode, time_interval, queue_dict[convert_to_name(cache_class=cache_class, param=current_cache_param)],
                          cache_class, cache_size, reader.__class__,
                          reader_params, current_cache_param))
        processes[p] = convert_to_name(cache_class=cache_class, param=current_cache_param)
        p.start()

    while finished < len(cache_classes):
        plot_now = False
        for name, q in queue_dict.items():
            while True:
                try:
                    hc_mc = queue_dict[name].get_nowait()
                    if sum(hc_mc) == 0:
                        hr_dict[name].append(hr_dict[name][-1])
                    else:
                        hr_dict[name].append(hc_mc[0] / sum(hc_mc))
                    plot_now = True
                except Empty:
                    time.sleep(5)
                    break

        if plot_now:
            _plot_ihrc(cache_classes, param_list, hr_dict, time_mode, figname)
            time.sleep(5)

        # check whether there are finished computations
        # for future, alg in future_to_alg.items():
        #     if future.done():
        process_to_remove = []
        for p, name in processes.items():
            if not p.is_alive():
                finished += 1
                while True:
                    try:
                        hc_mc = queue_dict[name].get_nowait()
                        if sum(hc_mc) == 0:
                            hr_dict[name].append(hr_dict[name][-1])
                        else:
                            hr_dict[name].append(hc_mc[0]/sum(hc_mc))
                    except Empty:
                        break
                queue_dict[name].close()
                del queue_dict[name]
                process_to_remove.append(p)
                print("{}/{} {} finished".format(finished, len(cache_classes), name))
                # print(", ".join(["{:.2f}".format(i) for i in hr_dict[name]]))

        for p in process_to_remove:
            del processes[p]
        process_to_remove.clear()

    _plot_ihrc(cache_classes, param_list, hr_dict, time_mode, figname, last_one=True)
    return hr_dict


def plot_IHR_with_dat(dat, **kwargs):
    import os
    reader = CsvReader(dat, init_params=AKAMAI_CSV3)
    cache_param = []
    ASig_prob_list = [0.8, 0.9999]
    for alg in kwargs.get("compare_algs", ()):
        if alg == "LRU":
            cache_param.append(None)
        elif alg == "Optimal" or alg == "ASigOPT":
            cache_param.append({"reader": reader})
        elif alg == "ASig0430":
            cache_param.append({"lifetime_prob": ASig_prob_list.pop()})
        else:
            raise RuntimeError("unknown alg {} {}".format(alg, kwargs))
    kwargs["cache_param"] = cache_param
    kwargs["figname"] = "0420/{}_{}.png".format(os.path.basename(dat), kwargs["cache_size"])
    plot_IHR(reader, **kwargs)


def run_akamai():
    from PyMimircache.utils.jobRunning import run_akamai3_parallel

    change_kwargs_list = [{"cache_size": 8000},
                          {"cache_size": 32000},
                          {"cache_size": 128000}
                          ]

    run_akamai3_parallel(plot_IHR_with_dat, fixed_kwargs={"time_mode": "v",
                                                 "time_interval": 1200000,
                                                 "compare_algs": ("LRU", "Optimal", "ASigOPT", "ASig0416", "ASig0416"),
                                                 },
                         change_kwargs_list=change_kwargs_list,
                         threads=8)

def run_data(dat):
    cache_size = 8000
    # cache_size = 32000
    # cache_size = 320000
    reader = CsvReader(dat, init_params=AKAMAI_CSV4)
    next_access_time = get_next_access_vtime(reader.file_loc, init_params=AKAMAI_CSV4)
    plot_IHR(reader, time_mode="v", time_interval=800000, cache_size=cache_size,
             compare_algs=("LRU", "Optimal", "ASig0416", "ASig0508", "LHD", "Hyperbolic"),
             cache_param=(None, {"reader": reader}, # {"reader": reader},
                          {"evict_type": "MRD", "lifetime_prob": 0.9999, "next_access_time": next_access_time},
                          {"next_access_time": next_access_time, "lifetime_prob": 0.9999},
                          {"update_interval": 200000, "coarsen_age_shift": 5, "n_classes":20, "max_coarsen_age":800000, "dat_name": "A"},
                          None,
                          ),
             figname="0601akamai4/{}_{}.png".format(os.path.basename(reader.file_loc), cache_size))



def get_next_access_vtime(dat, init_params):
    reader = CsvReader(dat, init_params=init_params)
    next_access_time = []

    if os.path.exists("{}.next_access_time.pickle".format(reader.file_loc)):
        print("no need to cal {}".format(reader.file_loc))
        with open("{}.next_access_time.pickle".format(reader.file_loc), "rb") as ifile:
            next_access_time = pickle.load(ifile)
    else:
        print("calculating next access time for {}".format(dat))
        next_access_dist = get_next_access_dist(reader)
        for t, req in enumerate(reader):
            if next_access_dist[t] == -1:
                next_access_time.append(-1)
            else:
                next_access_time.append(t + next_access_dist[t])
        reader.reset()
        with open("{}.next_access_time.pickle".format(reader.file_loc), "wb") as ofile:
            pickle.dump(next_access_time, ofile)
    return next_access_time

def t():
    from multiprocessing import Process
    pp = []
    for dat in [
        "/home/jason/ALL_DATA/akamai4/splitByType/lax.vod",
        "/home/jason/ALL_DATA/akamai4/splitByType/nyc.dl",
        "/home/jason/ALL_DATA/akamai4/splitByType/lax.c1_dl",
        "/home/jason/ALL_DATA/akamai4/splitByType/nyc.c1_dl",
        "/home/jason/ALL_DATA/akamai4/splitByType/sjc.ecomm",
        "/home/jason/ALL_DATA/akamai4/splitByType/lax.dl",
        "/home/jason/ALL_DATA/akamai4/splitByType/nyc.ecomm",
        "/home/jason/ALL_DATA/akamai4/splitByType/nyc.sv",
        "/home/jason/ALL_DATA/akamai4/splitByType/sjc.c1_dl",
        "/home/jason/ALL_DATA/akamai4/splitByType/lax.ecomm",
        "/home/jason/ALL_DATA/akamai4/splitByType/sjc.dl",
        "/home/jason/ALL_DATA/akamai4/splitByType/lax.sv",
    ]:
        p = Process(target=run_data, args=(dat, ))
        p.start()
        pp.append(p)
    for p in pp:
        p.join()




if __name__ == "__main__":
    from PyMimircache.bin.conf import AKAMAI_CSV3, AKAMAI_CSV4, CPHY_CSV
    from PyMimircache.bin.conf import WIKI_CSV
    from PyMimircache.profiler.cHeatmap import CHeatmap
    from PyMimircache.cacheReader.vscsiReader import VscsiReader
    from PyMimircache.cacheReader.csvReader import CsvReader
    from PyMimircache.profiler.profilerUtils import get_breakpoints
    from PyMimircache.profiler.utils.dist import *


    import pickle
    import os, sys

    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    reader = CsvReader("/scratch/jason/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1.sample.100", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1.sample.20", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai_7d_trace/nyc_1966", init_params=AKAMAI_CSV4)
    # reader = CsvReader("/home/jason/ALL_DATA/wiki_bench/wiki.all.csv.sort", init_params=WIKI_CSV)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.21.109.244.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.28.122.86.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.28.122.142.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.28.122.183.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.28.123.50.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.28.123.65.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.28.123.94.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.250.31.57.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.250.31.89.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/19.250.31.231.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.233.230.144.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.233.230.160.anon.1", init_params=AKAMAI_CSV3)

    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", init_params=AKAMAI_CSV3)
    # reader = VscsiReader("../../data/trace.vscsi")


    RUN_TYPE = "PARAMS"
    RUN_TYPE = "ALG"
    # RUN_TYPE = "SMALLTEST"
    RUN_TYPE = "STUDY"

    # t()
    # sys.exit()


    # run_akamai()
    # sys.exit(1)

    next_access_time = []

    SAMPLE_RATE = 1
    if "sample" in reader.file_loc:
        SAMPLE_RATE = 10

    cache_size = 8000 // SAMPLE_RATE
    # cache_size = 32000
    # cache_size = 320000
    time_interval = 800000 // SAMPLE_RATE // SAMPLE_RATE


    if RUN_TYPE == "ALG":
        next_access_time = get_next_access_vtime(reader.file_loc, init_params=AKAMAI_CSV3)
        # next_access_time = get_next_access_vtime(reader.file_loc, init_params=WIKI_CSV)


        plot_IHR(reader, time_mode="v", time_interval=time_interval, cache_size=cache_size,
                 compare_algs=("LRU", "Optimal", "ASig0508", "LHD",  "Hyperbolic"),
                 cache_param=(None, {"reader": reader},
                              # {"lifetime_prob": 0.80},
                              # {"evict_type": "MRD", "lifetime_prob": 0.8, "next_access_time": next_access_time, "n_rnd_evict_samples": 16},
                              # {"evict_type": "ED", "next_access_time": next_access_time, "n_rnd_evict_samples": 16}
                              # {"evict_type": "MRD", "next_access_time": next_access_time, "lifetime_prob": 0.9999},
                              {"next_access_time": next_access_time, "lifetime_prob": 0.9999},
                              # {"next_access_time": next_access_time, "lifetime_prob": 0.9999,  "sigmoid_func_mid": "arctan3"},
                              # {"next_access_time": next_access_time, "lifetime_prob": 0.9999,  "sigmoid_func_low": "arctan3", "sigmoid_func_mid": "arctan3"},
                              # {"next_access_time": next_access_time, "lifetime_prob": 0.9999},
                              {"update_interval": 200000//SAMPLE_RATE, "coarsen_age_shift": 5, "n_classes":20, "max_coarsen_age":800000//SAMPLE_RATE, "dat_name": "A"},
                              None,
                              # {"evict_type": "MRD", "next_access_time": next_access_time, "lifetime_prob": 0.9999, "sigmoid_func": "arctan", "freq_boundary": (4, 200)},
                              # {"evict_type": "MRD", "next_access_time": next_access_time, "lifetime_prob": 0.9999, "sigmoid_func": "arctan", "freq_boundary": (80, 20000)},
                              # {"next_access_time": next_access_time, "lifetime_prob": 0.9999, "sigmoid_func": "arctan", "fit_interval": 12*8},
                              ),
                 figname="{}_{}_{}Sample.png".format(os.path.basename(reader.file_loc), cache_size, time_interval))


    elif RUN_TYPE == "STUDY":
        # next_access_time = []
        next_access_time = get_next_access_vtime(reader.file_loc, init_params=AKAMAI_CSV3)

        # use_log = True
        # log_base = 1.2
        # params = {}
        # with open(reader.file_loc + ".fit.{}.{}.{}".format("arctan", use_log, log_base)) as ifile:
        #     params_arctan = json.load(ifile)
        # with open(reader.file_loc + ".fit.{}.{}.{}".format("arctan3", use_log, log_base)) as ifile:
        #     params_arctan3 = json.load(ifile)
        # with open(reader.file_loc + ".fit.{}.{}.{}".format("tanh2", use_log, log_base)) as ifile:
        #     params_tanh2 = json.load(ifile)

        plot_IHR(reader, time_mode="v", time_interval=time_interval, cache_size=cache_size,
                 compare_algs=("LRU", "Optimal", "ASig0508", "ASig0508", "ASig0508", "ASig0508"),
                 cache_param=(None, {"reader": reader},
                              # {"evict_type": "MRD", "next_access_time": next_access_time, "lifetime_prob": 0.9999},
                              {"next_access_time": next_access_time, "lifetime_prob": 0.9999, "use_log": False,
                               "options":[]},

                              {"next_access_time": next_access_time, "lifetime_prob": 0.9999,
                               "sigmoid_func_low": "arctan3", "sigmoid_func_mid": "arctan3", "use_log": False,
                               "options":[]},

                              {"next_access_time": next_access_time, "lifetime_prob": 0.9999,
                               "sigmoid_func_low": "min_arctan3", "sigmoid_func_mid": "min_arctan3", "use_log": False,
                               "options":[]},

                              {"next_access_time": next_access_time, "lifetime_prob": 0.9999,
                               "sigmoid_func_low": "min_arctan", "sigmoid_func_mid": "min_arctan", "use_log": False,
                               "options":[]},

                              # {"next_access_time": next_access_time, "lifetime_prob": 0.9999, "use_log": False,
                              #  "fixed_numerator": 1, "use_fifo": True},

                              # {"next_access_time": next_access_time, "lifetime_prob": 0.9999,
                              #  "sigmoid_func_low": "arctan3", "sigmoid_func_mid": "arctan3", "use_log": False,
                              #  "fixed_numerator": 1, "use_fifo": True},

                              # {"next_access_time": next_access_time, "lifetime_prob": 0.9999,
                              #  "sigmoid_func_low": "tanh2", "sigmoid_func_mid": "tanh2", "use_log": False, "fixed_numerator": 1},
                              ),
                 figname="{}_study_fifo2.png".format(os.path.basename(reader.file_loc)))


    elif "oneDay" in reader.file_loc:
        plot_IHR(reader, time_mode="v", time_interval=2000000, cache_size=200000, compare_algs=("LRU", "Optimal", "ASigOPT"),
                 cache_param=(None, {"reader": reader}, {"reader": reader}), figname="oneDayData_OPT.png")

    elif RUN_TYPE == "PARAMS":
        next_access_time = get_next_access_vtime(reader.file_loc, init_params=AKAMAI_CSV3)

        plot_IHR(reader, time_mode="v", time_interval=200000, cache_size=8000, alg="ASig0508",
                 compare_cache_params=(
                              {"next_access_time": next_access_time, "lifetime_prob": 0.6},
                              {"next_access_time": next_access_time, "lifetime_prob": 0.8},
                              {"next_access_time": next_access_time, "lifetime_prob": 0.9},
                              {"next_access_time": next_access_time, "lifetime_prob": 0.92},
                              {"next_access_time": next_access_time, "lifetime_prob": 0.94},
                              {"next_access_time": next_access_time, "lifetime_prob": 0.96},
                              {"next_access_time": next_access_time, "lifetime_prob": 0.9999}
                              # {"lifetime_prob": 0.9999, "freq_boundary": (8, 200)},
                              # {"lifetime_prob": 0.9999, "freq_boundary": (24, 200)},
                              # {"lifetime_prob": 0.9999, "log_base": 2.0},
                              # {"lifetime_prob": 0.9999, "log_base": 4.0},
                              # {"lifetime_prob": 0.9999, "fit_interval": 12 * 20},
                              # {"lifetime_prob": 0.9999}
                              ),
                 figname="paramTest.png"
                 # figname="lifetime_prob.png"
        )
    elif RUN_TYPE == "SMALLTEST":
        reader = VscsiReader("../../data/trace.vscsi")
        plot_IHR(reader, time_mode="v", time_interval=2000, cache_size=2000,
                 compare_algs=("LRU", "Optimal", "ASig0416", "ASig0430"),
                 cache_param=(None, {"reader": reader}, # {"reader": reader},
                              {"evict_type": "MRD", "next_access_time": [], "lifetime_prob": 0.9999, "freq_boundary": (12, 200), "log_base": 2},
                              {"evict_type": "MRD", "next_access_time": [], "lifetime_prob": 0.9999, "freq_boundary": (12, 200), "log_base": 2},
                              ),
                 figname="small_test.png")
