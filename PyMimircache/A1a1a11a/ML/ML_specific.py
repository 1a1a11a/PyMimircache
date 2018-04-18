# coding=utf-8

from PyMimircache.profiler.cHeatmap import get_breakpoints
import csv

import sys, time

sys.path.append("../")
sys.path.append("./")
from PyMimircache import *
from PyMimircache.bin.conf import initConf
from PyMimircache.A1a1a11a.ML.featureGenerator import *

################################## Global variable and initialization ###################################

TRACE_TYPE = "cloudphysics"
TRACE_DIR, NUM_OF_THREADS = initConf(TRACE_TYPE, trace_format='variable')
CUTOFF = 7
TRAIN_RATIO = 0.3

t_virtual = 2000  # 60 * 1000000
t_real = 60 * 1000000













def get_feature_specific(file_loc, time_interval=60*1000000, outfile="feature_specific.csv"):
    # if outfile[outfile.rfind('/')+2] == '0':
    #     print("ignore {}".format(outfile))
    #     return
    if os.path.exists(outfile):
        return
    reader = VscsiReader(file_loc)
    ts_dif = get_ts_difference(reader)
    rds = get_rd(reader)
    # rds = get_bd(reader)

    # frds = get_diff_next_access_time(reader)
    # frds = get_fd(reader)
    frds = get_frd(reader)

    freq = get_freq(reader)
    # bp = cHeatmap().gen_breakpoints(reader, 'r', time_interval=time_interval)
    bp = get_breakpoints(reader, 'v', time_interval=time_interval)
    freq2 = get_freq_interval(reader, bp)
    req_rate = get_request_rate_interval(reader, bp)
    cold_rate = get_cold_miss_rate_interval(reader, bp)

    pos = 0

    assert len(frds) == len(rds), "frds len {}, rds len {}".format(len(frds), len(rds))
    assert len(frds) == len(freq), "frds len {}, freq len {}".format(len(frds), len(freq))
    assert len(frds) == len(ts_dif), "frds len {}, ts_dif len {}".format(len(frds), len(ts_dif))
    assert len(freq2) == len(cold_rate), "freq2 len {}, cold_rate len {}".format(len(freq2), len(cold_rate))
    assert len(freq2) == len(req_rate), "freq2 len {}, req_rate len {}".format(len(freq2), len(req_rate))


    ######################################## SVM light use ########################################
    # maxY = np.max(frds)
    # frds[frds == -1] = maxY * 4
    # new_frds = [int(math.log2(i+2)/2-5) for i in frds]
    # frds = []
    # for i in new_frds:
    #     if i < 0:
    #         frds.append(0)
    #     else:
    #         frds.append(i)
    #
    # maxX = np.max(rds)
    # rds[rds == -1] = maxX * 4
    # new_rds = [int(math.log2(i+2)/2-5) for i in rds]
    # rds = []
    # for i in new_rds:
    #     if i < 0:
    #         rds.append(0)
    #     else:
    #         rds.append(i)

    ################################## End of SVM light use ###################################

    with open(outfile, 'w') as ofile:
        writer = csv.writer(ofile)
        for i in range(len(rds)):
            ################## filter out all the -1 in rd and frd ###################
            # if rds[i] != -1 and frds[i] != -1:
            #     # frd, rd, freq, real_time_diff
            #     row = [ int(math.log(frds[i]+1, 4) - 5), int(math.log(rds[i]+1, 4) - 5), freq[i], ts_dif[i]]
            #     for j in range(2):
            #         if row[j] < 0:
            #             row[j] = 0
            # elif rds[i] != -1:
            #     row = [-1, int(math.log(rds[i] + 1, 4) - 5), freq[i], ts_dif[i]]
            #     if row[1] < 0:
            #         row[1] = 0
            # elif frds[i] != -1:
            #     row = [int(math.log(frds[i] + 1, 4) - 5), -1, freq[i], ts_dif[i]]
            #     if row[0] < 0:
            #         row[0] = 0
            # else: # both -1
            #     row = [-1, -1, freq[i], ts_dif[i]]



            row = [frds[i], rds[i], freq[i], ts_dif[i]]


            ############## features about last/next rd, frd ###############
            # if i == 0:
            #     row.extend([0, rds[1], 0, frds[1]])
            #
            # elif i == len(rds)-1:
            #     row.extend([rds[len(rds)-2], 0, frds[len(rds)-2], 0])
            # else:
            #     row.extend([rds[i-1], rds[i+1], frds[i-1], frds[i+1]])
            # for j in range(4):
            #     if row[3+j] == -1:
            #         row[3+j] = 0

            ############## features about translating frd to cache size ############
            # if frds[i] == -1:
            #     row = [0]
            # elif frds[i] < CACHE_SIZE:
            #     row = [1]
            # else:
            #     row = [2]

            ############## split rd==-1 as new feature #####################
            # if rds[i] == -1:
            #     row.extend([1, 0])
            # else:
            #     row.extend([0, rds[i]])

            ################# add freq, req_rate, cold_miss_rate interval ##################
            row.extend([freq2[pos], req_rate[pos], cold_rate[pos]]) # , i])

            ################# add difference in freq, req_rate, cold_miss_rate interval ################
            if pos == 0:
                row.extend([freq2[pos], req_rate[pos], cold_rate[pos]])
            else:
                row.extend([freq2[pos] - freq2[pos-1],
                            req_rate[pos] - req_rate[pos-1],
                            cold_rate[pos] - cold_rate[pos-1]])

            writer.writerow(row)
            ######################################## SVM light use ########################################
            # -1 1:0.43 3:0.12 9284:0.2
            # s = "{}".format(row[0])
            # for j in range(1, len(row)):
            #     s += " {}:{:.4f}".format(j, row[j])
            #
            # ofile.write("{}\n".format(s))
            #################################### End of SVM light use ####################################

            if i == bp[pos+1]:
                pos += 1
                # print("i {} pos {}".format(i, pos))

    reader.close()


if __name__ == "__main__":
    dat = "../../data/trace.vscsi"
    trace_num = "test"
    t_virtual = 2000 # 60 * 1000000
    t_real = 60 * 1000000

    ################################ temporary generation functions ##################################
    # trace_num = "w101"
    # dat = "{}/{}_vscsi1.vscsitrace".format(TRACE_DIR, trace_num)

    t1 = time.time()
    get_feature_specific(dat, time_interval=t_virtual,
                         outfile="../features/specific_rd/{}_t{}.csv".format(trace_num, t_virtual))
    print(time.time() - t1)
    sys.exit(0)

    ################################## cal correlation coefficient ##################################
    # with Pool(cpu_count()) as p:
    #     p.map(rd_frd_coefficient,
    #           ["{}/{}".format(TRACE_DIR, f)
    #                        for f in os.listdir(TRACE_DIR) if f.endswith("vscsitrace")])
    # sys.exit(0)

    ################################## BATCH JOB: generate features #################################
    with ProcessPoolExecutor(4) as p:
        futures = {p.submit(get_feature_specific,
                            "{}/{}".format(TRACE_DIR, f),
                            t,
                            "features/specific_rd/{}_t{}".format(f[:f.find('_')], t_virtual)):
                       f for f in os.listdir(TRACE_DIR) if f.endswith("vscsitrace")}
        for future in as_completed(futures):
            print(futures[future])
