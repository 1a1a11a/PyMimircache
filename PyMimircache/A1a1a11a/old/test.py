# coding=utf-8

import os, time, csv, sys
import pickle
from collections import deque
from collections import defaultdict

from PyMimircache.cacheReader.binaryReader import BinaryReader

try:
    import matplotlib

    import PyMimircache.c_heatmap as c_heatmap
    import numpy as np
    from matplotlib import pyplot as plt

    from PyMimircache.cacheReader.csvReader import CsvReader
    from PyMimircache.cacheReader.vscsiReader import VscsiReader
    from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
    from PyMimircache.utils.printing import *
    from PyMimircache.profiler.cHeatmap import CHeatmap
    from PyMimircache.top.cachecow import Cachecow
    from PyMimircache.profiler.twoDPlots import *
    from PyMimircache.cacheReader.plainReader import PlainReader
    from PyMimircache import cGeneralProfiler
    from PyMimircache import generalProfiler
    from PyMimircache import *
    import PyMimircache.c_generalProfiler as c_generalProfiler
    from PyMimircache.cache.lru import LRU
    from PyMimircache.profiler.evictionStat import *
    import PyMimircache.c_LRUProfiler as c_LRUProfiler
except Exception as e:
    print(e)

DEFAULT_PARAM = {"max_support": 4,
                 "min_support": 1,
                 "confidence": 0,
                 "item_set_size": 20,
                 "prefetch_list_size": 2,
                 "cache_type": "LRU",
                 "sequential_type": 0,
                 "max_metadata_size": 0.1,
                 "block_size": 64 * 1024,
                 "sequential_K": 0,
                 "cycle_time": 2,
                 "AMP_pthreshold": 256
                }


def draw2d_here(l, **kwargs):
    if 'figname' in kwargs:
        filename = kwargs['figname']
    else:
        filename = '2d_plot.png'

    plt.plot(l, 'r-')

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
    INFO("plot is saved at the same directory")
    plt.clf()





def test1(dat, type, figname="w38_dat2_HRC_LFU.png"):
    c = Cachecow()
    if type == 'v':
        c.vscsi(dat)
    elif type == "p":
        c.open(dat)
    p = c.profiler("LFU", cache_size=1600000, num_of_threads=8)
    p.plotHRC(figname=figname, cache_size=1600000) #, auto_resize=True) #, cache_size=38000)

    c.diff_heatmap("v", 100000, "hit_rate_start_time_end_time", algorithm1="LRU", algorithm2='LFU',
                   figname="heatmap_" + figname, cache_size=20000)
    # c.heatmap("r", 10000000, "rd_distribution", algorithm="Optimal", cache_size=2000, num_of_threads=8)
    # c.twoDPlot("r", 1000000, "request_num", figname="request_num.png")


def test2(dat, type='v', figname="2LRU_HRC.png"):
    if type == 'v':
        reader = VscsiReader(dat)
    elif type == 'p':
        reader = PlainReader(dat)

    cG = cGeneralProfiler(reader, "LRU", 40000, bin_size=10, num_of_threads=8)
    cG.plotHRC(figname=figname)


def test3(dat):
    reader = VscsiReader(dat)
    p = generalProfiler(reader, "Optimal", 40000, bin_size=200, num_of_threads=8, cache_params={"reader": reader})
    p.plotHRC()


def test4(dat, mode, time_interval, cache_size, figname="LRU_err_rate.png"):
    reader = VscsiReader(dat)
    l = c_generalProfiler.get_err(reader.cReader, 'r',time_interval , cache_size, "LRU")
    print(len(l))
    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(l)))
    draw2d_here(l, figname=figname, xticks=xticks, xlabel='time({})'.format(mode),
           ylabel='error rate(interval={})'.format(time_interval),
           title='error rate 2D plot')




def test5(dat):
    TIME_INTERVAL = 10000000
    reader = VscsiReader(dat)
    h = CHeatmap()
    bp = get_breakpoints(reader, 'r', TIME_INTERVAL)
    bp_len = [bp[i]-bp[i-1] for i in range(1, len(bp))]
    print( "len: {}, max: {}, min: {}, ave: {}".format(len(bp_len), max(bp_len), min(bp_len), sum(bp_len)/len(bp_len)) )
    print(bp_len)
    ave = sum(bp_len)/len(bp_len)
    l1 = []
    l2 = []
    classification_list = []
    i = 0
    while i < len(bp_len):
        if i+3<len(bp_len) and bp_len[i] > ave and bp_len[i+1] > ave and bp_len[i+2] > ave and bp_len[i+3]>ave:
            classification_list.append(2)
            l2.append(i)
        elif i-3>=0 and bp_len[i] > ave and bp_len[i-1]>ave and bp_len[i-2]>ave and bp_len[i-3]>ave:
            l2.append(i)
            classification_list.append(2)
        else:
            l1.append(i)
            classification_list.append(1)

        i += 1

    set1 = set(l1)
    set2 = set(l2)

    bp_pos = 0
    num = 0



    # print(l1)
    # print(l2)
    print(classification_list)

    f1 = open("w38_dat1", 'w')
    f2 = open("w38_dat2", 'w')
    f = open("dat", 'w')

    for req in reader:
        num += 1
        if classification_list[bp_pos] == 1:
            f1.write(str(req) + '\n')
            f.write(str(req) + '\n')
        elif classification_list[bp_pos] == 2:
            f2.write(str(req) + '\n')
            f.write("12345\n")
        else:
            print("WHAT?")
        if num == bp_len[bp_pos]:
            num = 0
            bp_pos += 1

    f1.close()
    f2.close()
    f.close()


def test6_optimal():
    DAT = "../data/trace.vscsi"
    DAT= '../data/w03_vscsi1.vscsitrace'
    c = Cachecow()
    c.vscsi(DAT)
    p = c.profiler("Optimal", cache_size=6000000, bin_size=1200, num_of_threads=8)
    p.plotHRC("optimal3_2.png")



def test_mix_dat():
    DAT1 = 100
    DAT2 = 101
    DAT3 = 102
    r1 = VscsiReader("../data/traces/w{}_vscsi1.vscsitrace".format(DAT1))
    r2 = VscsiReader("../data/traces/w{}_vscsi1.vscsitrace".format(DAT2))
    r3 = VscsiReader("../data/traces/w{}_vscsi1.vscsitrace".format(DAT3))
    readers = [r1, r2, r3]

    import random


    with open("mix_dat.txt", 'w') as ofile:
        while len(readers):
            r = random.choice(readers)
            try:
                x = next(r)
                ofile.write("{}\n".format(x))
                # print("{}".format(readers.index(r)))
            except:
                readers.remove(r)


def test_following(filename):
    RANGE = 8

    from collections import defaultdict
    # r1 = vscsiReader("/home/cloudphysics/traces/{}".format(filename))
    r1 = VscsiReader("../data/traces/{}".format(filename))
    d = defaultdict(list)
    last = None
    for e in r1:
        if last:
            d[last].append(e)
        last = e

    r1.reset()

    counters = [0] * (RANGE+1)
    counter_out = 0
    for k, v in d.items():
        l = len(set(v))-1
        n = len(v)-1
        if l <= RANGE:
            counters[l] += n
        else:
            counter_out += n
            counters[-1] += n
            # print("out {}".format(l))

    # l = [len(set(i)) for i in d.values()]

    print("total {}, unique {}".format(r1.get_num_of_req(), len(d)))
    print(counters)
    # print(l)

    # print("out {}".format(counter_out))
    plt.bar(range(1, RANGE+2, 1), counters, align="center")
    plt.xlim(RANGE+2)
    plt.title("distribution of count of different requests immediately follows")
    plt.xlabel("count of different requests immediately follows")
    plt.ylabel("number of requests")
    xticks = [i for i in range(0, RANGE+1, 1)]
    xticks.append("{}+".format(RANGE))
    plt.xticks(range(RANGE+1), xticks)
    plt.savefig("0916_distribution/" + str(filename)+"_following_distribution.png", dpi=600)
    plt.clf()


def testa(dat):
    r = VscsiReader("../A1a1a11a/prefetch_input_data/{}/xab".format("w90"))
    r = CsvReader(dat, init_params={"real_time_column": 1, "label_column": 5})
    # r = vscsiReader("/scratch/A1a1a11a/traces/{}".format(dat))
    n = r.get_num_of_req()
    nu = r.get_num_of_uniq_req()

    counter = 0
    d = {}

    last = None

    print("total {}, unique {}".format(n, nu))
    for i in r:
        if not last:
            last = i
            continue

        if last in d and d[last] == i:
            counter += 1
        d[last] = i
        last = i

    percent = counter / (n - nu)
    print("counter {}, percentage {}".format(counter, percent))


def testb0(dat):
    if os.path.exists("/scratch/A1a1a11a/traces/pickle/{}.pickle".format(dat[:dat.rfind(".")])):
        return
    r = VscsiReader("/scratch/A1a1a11a/traces/{}".format(dat))
    l = []
    n = r.get_num_of_req()
    nu = r.get_num_of_uniq_req()

    for i in r:
        l.append(i)
    with open("/scratch/A1a1a11a/traces/pickle/{}.pickle".format(dat[:dat.rfind(".")]), 'wb') as ofile:
        pickle.dump([n, nu, l], ofile)



def testb(dat):
    size = 200


    t1 = time.time()
    ofile = open("simple prefetch analysis.out", 'a')
    writer = csv.writer(ofile)
    # writer.writerow(["total", "unique", "distriubtion total "+str(size), "distribution 1", "distribution 2", "distribution 3"])


    # r = vscsiReader("../A1a1a11a/prefetch_input_data/{}/xab".format("w90"))
    # r = vscsiReader("../data/trace.vscsi")
    # r = vscsiReader("/scratch/A1a1a11a/traces/{}".format(dat))
    # n = r.get_num_of_req()
    # nu = r.get_num_of_uniq_req()


    last_dict = {}
    l = []
    result = [0] * size
    last = 0        # or "begin", just to reduce an if

    # for i in r:
    #     l.append(i)


    ifile = open("/scratch/A1a1a11a/traces/pickle/{}".format(dat), 'rb')
    print("exist {}".format(os.path.exists("/scratch/A1a1a11a/traces/pickle/{}".format(dat))))
    n, nu, l = pickle.load(ifile)


    print("total {}, unique {}, {}s".format(n, nu, time.time() - t1))
    t1 = time.time()
    for i in range(len(l)):
        if last in last_dict:
            e = last_dict[last]
            for j in range(0, size, 1):
                if i+j >= len(l):
                    break
                if l[i+j] == e:
                    result[j] += 1
                    break

        last_dict[last] = l[i]
        last = l[i]
    # print(last_dict)


    temp = [i/(n - nu) for i in result]
    print("distribution {}, {}s".format(temp[:5], time.time() - t1))
    row = [dat, n, nu, sum(temp)]
    row.extend(temp)
    writer.writerow(row)
    ofile.flush()


def test_appear_times(dat):
    # r = vscsiReader("/scratch/A1a1a11a/traces/{}".format(dat))
    r = VscsiReader("../data/trace.vscsi")
    d = defaultdict(int)
    for i in r:
        d[i] += 1

    d2 = defaultdict(int)
    for k,v in d.items():
        d2[v] += 1

    max_v = max(d2.keys())
    y = [0] * (max_v+1)
    for k, v in d2.items():
        y[k] = v


    plt.loglog(range(max_v+1), y, linewidth=3.0, basex=2, basey=2)
    plt.savefig("request_num_distribution.png", dpi=1200)


def test0930():
    SET_SIZE = 20
    CACHE_SIZE = 2000
    reader = VscsiReader("../data/trace.vscsi")
    # reader = csvReader("../profiler/MSR/"+"hm_0.csv", init_params={"real_time_column": 1, "label_column": 5})

    cache  = LRU(cache_size=CACHE_SIZE)
    de_in  = deque()
    de_out = deque()
    fingerprintDict = defaultdict(list)
    l = []

    for r in reader:
        if r in cache:
            cache._update(r, )
        else:
            # if len(de_in) == SET_SIZE:
            fingerprintDict[r].append((0, set(de_in)))         # 0 means in, 1 means out
            cache._insert(r, )
            if len(cache) > CACHE_SIZE:
                r_evicted = cache.evict()
                # if len(de_out) == SET_SIZE:
                if len(fingerprintDict[r_evicted])!=0:
                    assert (fingerprintDict[r_evicted][-1][0] == 0), "why? "
                fingerprintDict[r_evicted].append( (1, set(de_out)) )

                de_out.append(r_evicted)
                if len(de_out) > SET_SIZE:
                    de_out.popleft()

        de_in.append(r)
        if len(de_in) > SET_SIZE:
            de_in.popleft()

    with open("fingerprints.out", 'wb') as ofile:
        pickle.dump(fingerprintDict, ofile)

def test0930_2():
    with open("fingerprints.out", 'rb') as ifile:
        fingerprintDict = pickle.load(ifile)

    counter_len = 0
    counter = 0
    max_length = -1
    l = []
    for k,v in fingerprintDict.items():
        if len(v) == 1:
            continue
        counter += 1
        counter_len += len(v)
        if len(v) > max_length:
            max_length = len(v)
        l.append(len(v))

    plt.yscale('log', nonposy='clip')
    plt.hist(l)
    plt.savefig("hist.png")


    print("dict length {}, ave length {}, max {}".format(
                    len(fingerprintDict), counter_len/counter, max_length))


def test0930_3():
    with open("fingerprints.out", 'rb') as ifile:
        fingerprintDict = pickle.load(ifile)

    for k,v in fingerprintDict.items():
        if len(v) == 1 or len(v) <= 4:
            continue
        intersections = set.intersection(*[s[1] for s in v])
        unions = set.union(*[s[1] for s in v])
        print("{}: list size {}".format(k, len(v)))
        for i in range(len(v)//2):
            assert v[2*i][0]+ v[2*i+1][0] == 1, "what is the value? {}".format(v[2*i][0]+ v[2*i+1][0])
            intersections = set.intersection(v[2*i][1], v[2*i+1][1])
            unions = set.union(v[2*i][1], v[2*i+1][1])
            print("intersection size {}, union size {}, {}".format(len(intersections), len(unions), sorted(intersections)))
        print(" ")
        # print("{}: list size {}, intersection size {}, union size {}, {}".format(
        #         k, len(v), len(intersections), len(unions), sorted(intersections)))


def test1107():
    reader = VscsiReader("{}/trace.vscsi".format("../data"))
    cH = CHeatmap()
    bpr = get_breakpoints(reader, 'r', 1000000)
    bpv = get_breakpoints(reader, 'v', 1000)

    cH.heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
    cH.heatmap(reader, 'r', 10000000, "rd_distribution", num_of_threads=8)
    cH.heatmap(reader, 'r', 10000000, "future_rd_distribution", num_of_threads=8)
    cH.heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", algorithm="FIFO", num_of_threads=8, cache_size=2000)
    cH.diff_heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", cache_size=2000,
                    algorithm1="LRU_K", algorithm2="Optimal", cache_params1={"K": 2},
                    cache_params2=None, num_of_threads=8)


def test1129():
    dat = "../data/traces/w41_vscsi1.vscsitrace"
    reader = VscsiReader(dat)
    n = reader.get_num_of_req()
    print("total {} requests".format(n))
    # for o, i in enumerate(reader):
    #     if (o > n - 100):
    #         print("{}: {}".format(o, i))
    hr = c_generalProfiler.get_hit_rate(reader.cReader, "Optimal", 100000, bin_size=10000)
    print(hr)


def test0129(dat):
    c = Cachecow()
    c.vscsi(dat)
    c.plotHRCs(["LRU", "mimir"], cache_params=[None, DEFAULT_PARAM],
               figname="test.png", num_of_threads=8,
               cache_size=50000, bin_size=2500, auto_resize=False)


def test01292(dat):
    c = Cachecow()
    c.open(dat, data_type='l')
    c.plotHRCs(["mimir"], cache_params=[DEFAULT_PARAM],
               figname="test.png", num_of_threads=8,
               cache_size=2000, bin_size=2000, auto_resize=False)

def test0222(dat):
    with BinaryReader("/scratch/A1a1a11a/20161001.sort.mapped.bin.LL", "<LL") as r:
        count = 0
        line = r.read()
        print(line)
        while line:
            count += 1
            line = r.read()
            if count %100000 == 0:
                print(line)
            # if count > 10000:
            #     break
        print(line)
        print(len(r))
    print(count)

def test0310():
    print("hello")

    c = Cachecow()
    c.csv("../CExtension/837", init_params={"header":False, "delimiter":"\t", "label_column":5, 'real_time_column':1})
    print("self close")
    print(c.reader)
    c.close()
    print("hello done")
    del c
    print("del done")

def test0321(dat):
    reader = VscsiReader(dat)
    rd = CLRUProfiler(reader).get_future_reuse_distance()
    maxRD = max(rd)
    l_distr = [0] * (maxRD+1)
    for r in rd:
        l_distr[r] += 1
    for n, l in enumerate(l_distr):
        if l>100000:
            print("{}: {}".format(n, l))
    plt.plot(l_distr)
    plt.savefig("rd_distribution/rd_distribution_{}.png".format(dat[dat.rfind('/')+1 : dat.rfind('_')]), dpi=600)
    plt.clf()
    plt.plot(l_distr[:-1])
    plt.savefig("rd_distribution/rd_distribution2_{}.png".format(dat[dat.rfind('/')+1 : dat.rfind('_')]), dpi=600)
    plt.xscale('log')
    plt.savefig("rd_distribution/rd_distribution2log_{}.png".format(dat[dat.rfind('/')+1 : dat.rfind('_')]), dpi=600)
    plt.clf()

def test03212():
    DATA_PATH = "/root/disk2/ALL_DATA/traces/"
    for f in os.listdir(DATA_PATH):
        if os.path.exists("rd_distribution/rd_distribution_{}.png".format(f[f.rfind('/')+1 : f.rfind('_')])):
            continue
        if 'vscsitrace' not in f:
            continue
        print(f)
        test0321("{}/{}".format(DATA_PATH, f))


def test0322(dat):
    with VscsiReader(dat) as r:
        rd = CLRUProfiler(r).get_reuse_distance()
    # new_rd = [rd[i] for i in range(len(rd)) if i%100==0 and rd[i]!=-1]
    new_rd1 = []
    new_rd2 = []
    new_rd = [new_rd1, new_rd2]

    new_rd1.append(rd[0])
    last_rd_list = 0
    last_rd = rd[0]
    for i in rd[1:]:
        if (i  < 8192):
            new_rd[0].append(i)
            last_rd = i
        else:
            # last_rd_list = (last_rd_list+1)%2
            new_rd[1].append(i)
            last_rd = i

    plt.plot(new_rd[0])
    plt.savefig("test.png", dpi=600)
    plt.clf()

def test0324():
    """
    this is the init combine splitted trace func for Akamai 
    :return: 
    """
    from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
    import heapq
    DIR = "/root/disk2/ALL_DATA/Akamai"

    for folder in os.listdir(DIR):
        if os.path.isdir("{}/{}".format(DIR, folder)):
            if folder == 'binary':
                continue
            print("current folder {}".format(folder))
            if not os.path.exists("{}/binary/{}/".format(DIR, folder)):
                os.makedirs("{}/binary/{}/".format(DIR, folder))
            writer_complete = TraceBinaryWriter(fmt="<LL", ofilename="{}/binary/{}/complete".format(DIR, folder))
            dict_complete = {}
            readers = []
            writers = []
            dicts   = []
            heap    = []
            for f in os.listdir("{}/{}".format(DIR, folder)):
                if not os.path.isfile("{}/{}/{}".format(DIR, folder, f)):
                    continue
                readers.append(CsvReader("{}/{}/{}".format(DIR, folder, f),
                                         init_params={"label_column": 5, "real_time_column": 1, "delimiter": "\t"}))
                writers.append(TraceBinaryWriter("{}/binary/{}/{}".format(DIR, folder, f), fmt="<LL"))
                dicts.append(defaultdict(int))
                r = readers[-1].read_time_req()
                # new_label = dicts[-1].get(r[1], len(dicts[-1]))
                # dicts[-1][r[1]] = new_label
                # writers[-1].write((r[0], new_label))
                heapq.heappush(heap, (int(r[0]), r[1], len(readers)-1))

            while len(heap):
                # print("heap length {}".format(len(heap)))
                r = heapq.heappop(heap)
                idx = r[2]

                new_label = dicts[idx].get(r[1], )
                label_complete = dict_complete.get(r[1], len(dict_complete))
                dicts[idx][r[1]] = new_label
                dict_complete[r[1]] = label_complete

                writers[idx].write((int(r[0]), new_label))
                writer_complete.write((int(r[0]), label_complete))

                r = readers[idx].read_time_req()
                if r:
                    heapq.heappush(heap, (int(r[0]), r[1], idx))

            for reader in readers:
                reader.close()
            for writer in writers:
                writer.close()
            writer_complete.close()


def test0325():
    readerc = PlainReader("../data/trace.txt", data_type='c')
    readerl = PlainReader("../data/trace.txt", data_type='l')
    readerv = VscsiReader("../data/trace.vscsi")

    l1 = c_LRUProfiler.get_reuse_dist_seq(readerc.cReader)
    l2 = c_LRUProfiler.get_reuse_dist_seq(readerl.cReader)
    l3 = c_LRUProfiler.get_reuse_dist_seq(readerv.cReader)
    print(l1[1260:1280])
    print(l2[1260:1280])
    print(l3[1260:1280])
    # for n, p in enumerate(zip(l1, l2, l3)):
    #     if p[0] != p[1] or p[0] != p[2]:
    #         print("not same {}:{}:{}, line {}".format(p[0], p[1], p[2], n))
    # else:
    #     print("all good")

def test03252():
    c = Cachecow()
    c.open("../data/trace.txt")
    for i in c:
        print(i)




if __name__ == "__main__":
    import time
    t1 = time.time()
    # sys.stdout = open("test2_2.out", 'w')
    # test0930()
    # test0930_2()
    # test0930_3()
    # test1129()
    # test0129("/scratch/A1a1a11a/traces/w94_vscsi1.vscsitrace")
    # test01292("/home/A1a1a11a/ALL_DATA/cloudphysics_txt_64K/w94.txt")
    # test0222(None)
    # test0310()
    # test0321("../data/trace.vscsi")
    # test0321("/root/disk2/ALL_DATA/traces/w105_vscsi1.vscsitrace")
    test03212()
    # test0322("../data/trace.vscsi")
    # test03252()