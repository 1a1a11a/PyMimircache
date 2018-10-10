# coding=utf-8

import os, sys, time, pickle, re
sys.path.append("../")
sys.path.append("./")
from PyMimircache import *
from PyMimircache.bin.conf import *
from multiprocessing import Pool
from functools import partial

################################## Global variable and initialization ###################################

TRACE_TYPE = "Akamai4"
# TRACE_TYPE = "cphy"
TRACE_FORMAT = "variable"
# TRACE_TYPE, TRACE_FORMAT = "MSR", "variable"

TRACE_DIR, NUM_OF_THREADS = initConf(TRACE_TYPE, trace_format=TRACE_FORMAT)
NUM_OF_THREADS = 48


########################################### main functions ##############################################


TIME_INTERVAL_r_AKA = 1200
TIME_INTERVAL_r_CPHY = 100 * 1000000
TIME_INTERVAL_r_MSR = 3600 * 1000000

GRANULARITY_CPHY = 1 * 1000000
GRANULARITY_AKA = 1
GRANULARITY_MSR =20 * 1000000

CACHE_SIZE = 8000

TIME_INTERVAL_v = 1 * 1000000  # 1000000


if 'akamai' in TRACE_TYPE.lower():
    TIME_INTERVAL_r = TIME_INTERVAL_r_AKA
    GRANULARITY = GRANULARITY_AKA

elif 'cphy' in TRACE_TYPE.lower():
    TIME_INTERVAL_r = TIME_INTERVAL_r_CPHY
    GRANULARITY = GRANULARITY_CPHY

elif "msr" in TRACE_TYPE.lower():
    TIME_INTERVAL_r = TIME_INTERVAL_r_MSR
    GRANULARITY = GRANULARITY_MSR



def run(dat, fig_folder="fig"):
    cache_size = CACHE_SIZE
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    dat_ID = os.path.basename(dat)
    dat_ID = re.split('[.]', dat_ID)[0]
    c = Cachecow()
    # c.binary(dat, init_params=AKAMAI_BIN_MAPPED2)

    if 'akamai3' in TRACE_TYPE.lower():
        c.csv(dat, init_params=AKAMAI_CSV3)
    elif 'akamai4' in TRACE_TYPE.lower():
        c.csv(dat, init_params=AKAMAI_CSV4)
    elif 'cphy' in TRACE_TYPE.lower():
        c.vscsi(dat)
    elif "msr" in TRACE_TYPE.lower():
        c.csv(dat, init_params=MSR_CSV)
    else:
        print("unknow trace type")
        return

    n_req = c.reader.get_num_of_req()
    if n_req < cache_size:
        cache_size = n_req // 3

    # c.csv(dat, init_params={"label": 2, "real_time": 1, "delimiter": "\t"})
    print("dat: {}, output {} {} req".format(dat, fig_folder, n_req))



    # trace stat
    if not os.path.exists("{}/{}.stat".format(fig_folder, dat_ID)):
        print("trace stat")
        with open("{}/{}.stat".format(fig_folder, dat_ID), "w") as ofile:
            ofile.write(c.stat())

    # request rate 2D plot
    if not os.path.exists("{}/{}_request_rate_r_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_r)):
        print("req rate plot")
        c.twoDPlot("request_rate", time_mode='r', time_interval=TIME_INTERVAL_r,
                   figname="{}/{}_request_rate_r_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_r))

    # mapping plot
    if not os.path.exists("{}/{}_mapping_overall.png".format(fig_folder, dat_ID)):
        print("mapping plot")
        c.twoDPlot("mapping", figname="{}/{}_mapping.png".format(fig_folder, dat_ID))

    # cold miss 2D plot
    if not os.path.exists("{}/{}_cold_miss_count_v_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_v)):
        print("cold miss plot")
        c.twoDPlot("cold_miss_count", time_mode='v', time_interval=TIME_INTERVAL_v,
          figname="{}/{}_cold_miss_count_v_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_v))
    if not os.path.exists("{}/{}_cold_miss_count_r_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_r)):
        c.twoDPlot("cold_miss_count", time_mode='r', time_interval=TIME_INTERVAL_r,
          figname="{}/{}_cold_miss_count_r_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_r))

    if not os.path.exists("{}/{}_cold_miss_ratio_v_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_v)):
        c.twoDPlot("cold_miss_ratio", time_mode='v', time_interval=TIME_INTERVAL_v,
                   figname="{}/{}_cold_miss_ratio_v_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_v))
    if not os.path.exists("{}/{}_cold_miss_ratio_r_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_r)):
        c.twoDPlot("cold_miss_ratio", time_mode='r', time_interval=TIME_INTERVAL_r,
                   figname="{}/{}_cold_miss_ratio_r_{}.png".format(fig_folder, dat_ID, TIME_INTERVAL_r))

    # popularity
    if not os.path.exists("{}/{}_popularity_2d.png".format(fig_folder, dat_ID)):
        print("popularity")
        c.twoDPlot("popularity", logX=True, logY=True,
                   figname="{}/{}_popularity_2d.png".format(fig_folder, dat_ID))


    # rd_distribution
    if not os.path.exists("{}/{}_rd_popularity_2d.png".format(fig_folder, dat_ID)):
        print("rd_popularity")
        c.twoDPlot("rd_popularity", logX=True, logY=False, cdf=True, # granularity=200,
                   figname="{}/{}_rd_popularity_2d.png".format(fig_folder, dat_ID))

    # rt_distribution
    if not os.path.exists("{}/{}_rt_popularity_2d.png".format(fig_folder, dat_ID)):
        print("rt_popularity")
        c.twoDPlot("rt_popularity", logX=True, logY=False, cdf=True, granularity=GRANULARITY,
                   figname="{}/{}_rt_popularity_2d.png".format(fig_folder, dat_ID))


    # eviction stat
    # c.evictionPlot('r', TIME_INTERVAL_r, "freq", "Optimal", cache_size,
    #   figname="{}_evic_freq_OPT_r_{}.png".format(figname_prefix, TIME_INTERVAL_r))
    # c.evictionPlot('r', TIME_INTERVAL_r, "accumulative_freq", "Optimal", cache_size,
    #   figname="{}_evic_accu_freq_OPT_r_{}.png".format(figname_prefix, TIME_INTERVAL_r))
    # c.evictionPlot('r', TIME_INTERVAL_r, "reuse_dist", "Optimal", cache_size,
    #   figname="{}_evic_rd_OPT_r_{}.png".format(figname_prefix, TIME_INTERVAL_r))


    # rd heatmap
    if not os.path.exists("{}/{}_rd_distribution.png".format(fig_folder, dat_ID)):
        print("heatmap plot")
        c.heatmap('r', "rd_distribution", time_interval=TIME_INTERVAL_r,
                  figname="{}/{}_rd_distribution.png".format(fig_folder, dat_ID),
                  num_of_threads=NUM_OF_THREADS)


    # hit rate curve
    if not os.path.exists("{}/{}_HRC_autoSize.png".format(fig_folder, dat_ID)):
        print("hit rate curve")
        c.plotHRCs(["LRU", "LFUFast", "ARC", "SLRU", "Optimal"],  # , "LRU_2"
                   [None, None, None, {"N":2}, None],
                   cache_size=cache_size, bin_size=cache_size//NUM_OF_THREADS//2+1,
                   auto_resize=True, num_of_threads=NUM_OF_THREADS,
                   # use_general_profiler=True,
                    save_gradually=True,
                   figname="{}/{}_HRC_autoSize.png".format(fig_folder, dat_ID))

    # real time

    if not os.path.exists("{}/{}_heatmap_LRU_r.png".format(fig_folder, dat_ID)):
        c.heatmap('r', "hr_st_et", algorithm="LRU",
                  time_interval=TIME_INTERVAL_r,
                  cache_size=cache_size, num_of_threads=NUM_OF_THREADS,
                  figname="{}/{}_heatmap_LRU_r.png".format(fig_folder, dat_ID))

    if not os.path.exists("{}/{}_diff_heatmap_LRU_OPT_r.png".format(fig_folder, dat_ID)):
        c.diff_heatmap('r', "hr_st_et",
                       time_interval=TIME_INTERVAL_r, algorithm1="LRU", algorithm2="Optimal",
                       cache_size=cache_size, num_of_threads=NUM_OF_THREADS,
                       figname="{}/{}_diff_heatmap_LRU_OPT_r.png".format(fig_folder, dat_ID))

    # logical time
    c.heatmap('v', "hr_st_et", algorithm="LRU",
              time_interval=TIME_INTERVAL_v,
              cache_size=cache_size, num_of_threads=NUM_OF_THREADS,
              figname="{}/{}_heatmap_LRU_v.png".format(fig_folder, dat_ID))
    c.diff_heatmap('v', "hr_st_et",
                           algorithm1="LRU", algorithm2="Optimal", time_interval=TIME_INTERVAL_v,
                           cache_size=cache_size, num_of_threads=NUM_OF_THREADS,
                           figname="{}/{}_diff_heatmap_LRU_OPT_v.png".format(fig_folder, dat_ID))


def run_msr():
    # Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime
    for f in os.listdir(TRACE_DIR):
        print(f)
        if ".txt" in f or ".sh" in f:
            continue
        run("{}/{}".format(TRACE_DIR, f), fig_folder="msr_fig")


def run_akamai2_0507():
    trace_dir = "/home/jason/ALL_DATA/akamai_new_logs"
    for f in os.listdir(trace_dir):
        if 'anon' in f:
            print(f)
            run("{}/{}".format(trace_dir, f), fig_folder="akamai2_fig")


def run_akamai3_0723():
    trace_dir = "/home/jason/ALL_DATA/akamai3/layer/1/"
    for f in os.listdir(trace_dir):
        if 'anon' in f:
            print(f)
            run("{}/{}".format(trace_dir, f), fig_folder="akamai3_fig")

def run_akamai4_0530():
    trace_dir = "/home/jason/ALL_DATA/akamai4/"
    for f in reversed(os.listdir(trace_dir)):
        if '.' not in f:
            print(f)
            run("{}/{}".format(trace_dir, f), fig_folder="akamai4_fig")



def run_ATC_SLIDE_0508():
    trace_dir = "/home/jason/ALL_DATA/SLIDE_ATC/parda"
    for f in os.listdir(trace_dir):
        if 'parda' in f:
            print(f)
            run("{}/{}".format(trace_dir, f), fig_folder="SLIDE_fig")


def run_akamai3_split_0824(output_folder="0824akamai3_split/"):
    trace_dir = "/home/jason/ALL_DATA/akamai3/layer/1/traceLength/"
    for i in range(1, 9):
        if not os.path.exists(output_folder + str(i)):
            os.makedirs(output_folder + str(i))
        for f in os.listdir(trace_dir + str(i)):
            run(trace_dir+str(i)+"/"+f, fig_folder=output_folder+str(i))


def run_akamai3_split_1005(output_folder="1005akamai3_clean/"):
    trace_dir = "/home/jason/ALL_DATA/akamai3/layer/1/clean0922/"
    if not os.path.exists("{}/oneDay".format(output_folder)):
        os.makedirs("{}/oneDay".format(output_folder))
        os.makedirs("{}/twoDay".format(output_folder))

    # for length in ["oneDay", "twoDay"]:
    #     for f in os.listdir("{}/{}".format(trace_dir, length)):
    #         run(trace_dir + "/{}/".format(length)+f, fig_folder=output_folder+length)

    run(trace_dir+"oneDayData.sort", fig_folder=output_folder)
    run(trace_dir+"twoDayData.sort", fig_folder=output_folder)


def run_cphy1_0910(output_folder="0910cphy1"):
    dat_folder = "/home/cloudphysics/traces"

    pfunc = partial(run, fig_folder=output_folder)
    dat_list = []

    for f in os.listdir(dat_folder):
        dat_list.append("{}/{}".format(dat_folder, f))
        # run("{}/{}".format(dat_folder, f), fig_folder=output_folder)
    with Pool(24) as pool:
        pool.map(pfunc, dat_list)

def run_cphy_1005(output_folder="1005cphy"):
    dat_folder = "/home/cloudphysics/traces"
    run("{}/w106_vscsi1.vscsitrace".format(dat_folder), fig_folder=output_folder)
    run("{}/w92_vscsi1.vscsitrace".format(dat_folder), fig_folder=output_folder)
    run("{}/w78_vscsi1.vscsitrace".format(dat_folder), fig_folder=output_folder)


if __name__ == "__main__":
    # run_akamai_0515()
    # run_akamai2_0507()
    # run_akamai3_0723()
    # run_akamai3_split_0824()
    # run_ATC_SLIDE_0508()
    # run_cphy1_0910()

    print("using {} threads".format(NUM_OF_THREADS))
    # run_msr()
    run_akamai4_0530()
    # run("../../data/trace.vscsi")