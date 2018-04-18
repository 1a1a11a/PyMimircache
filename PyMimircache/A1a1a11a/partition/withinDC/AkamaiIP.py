# coding=utf-8
"""
this file plot HRC of nodes in the same data center and a combined HRC 
the combined HRC is obtained by merging the splitted data into a combined dataset 
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

from PyMimircache import *
from PyMimircache.future.multiTraceProfiler import multiTraceProfiler


def run(folder):
    readers = []
    for f in os.listdir(folder):
        if not os.path.isfile("{}/{}".format(folder, f)):
            continue
        if f == 'complete':
            continue
        readers.append(BinaryReader("{}/{}".format(folder, f),
                                    init_params={"label":1, "real_time": 2, "fmt": "<LL"}))
        # readers.append(csvReader("{}/{}".format(folder, f),
        #                             init_params={"label_column": 5, "real_time_column": 1, "delimiter": "\t"}))
    mtp = multiTraceProfiler(readers, "LRU", cache_size=1000000)
    mtp.plotHRCs(save=False)
    mtp.plotHRCs(op="+", mix_mode="real_time",
                 figname="HRC_{}.pdf".format(folder[folder.rfind("/")+1:]))


def batch_mjolnir():
    """
    run multitrace profiler sequentially 
    :return: 
    """
    TRACE_DIR = "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/binary"
    for folder in os.listdir("{}".format(TRACE_DIR)):
        if folder != 'useless':
            run("{}/{}".format(TRACE_DIR, folder))
            print("{} finished".format(folder))

def batch_mjolnir_multithread():
    """
    run multitracce profiler using multithreading 
    :return: 
    """
    TRACE_DIR = "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv"
    with ProcessPoolExecutor(max_workers=40) as p:
        results = {p.submit(run, "{}/{}".format(TRACE_DIR, folder)):folder
                   for folder in os.listdir("{}".format(TRACE_DIR)) if folder!='useless'}
        for future in as_completed(results):
            print("{} finished".format(results[future]))


if __name__ == "__main__":
    # run("/root/disk2/ALL_DATA/Akamai/binary/1010")
    run("/root/disk2/ALL_DATA/Akamai/binary/1392")
    # run("/root/disk2/ALL_DATA/Akamai/dataCenterSplitted/1010")
    # batch_mjolnir()
    # batch_mjolnir_multithread()
