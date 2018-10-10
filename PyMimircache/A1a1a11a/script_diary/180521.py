"""
    ASigOffline


"""


import os, sys, math, time
import subprocess
from collections import defaultdict
from PyMimircache import CsvReader, VscsiReader
from PyMimircache.bin.conf import AKAMAI_CSV3, AKAMAI_CSV4
from PyMimircache.A1a1a11a.script_diary.sigmoid import sigmoid_fit
from PyMimircache.A1a1a11a.script_diary.sigmoid import *
import json, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed


DATA_DIR = "/home/jason/ALL_DATA/akamai3/layer/1/"

def sample_trace(sample_rate=20):
    dat = ["185.232.99.68.anon.1", "185.232.99.85.anon.1", "185.233.229.170.anon.1", "185.233.230.144.anon.1",
           "185.233.230.160.anon.1", "185.233.230.37.anon.1", "185.233.230.44.anon.1", "185.233.230.88.anon.1",
           "185.233.230.94.anon.1", "19.21.108.123.anon.1", "19.21.108.25.anon.1", "19.21.108.77.anon.1",
           "19.21.109.171.anon.1", "19.21.109.244.anon.1", "19.21.109.4.anon.1", "19.250.31.171.anon.1",
           "19.250.31.22.anon.1", "19.250.31.231.anon.1", "19.250.31.243.anon.1", "19.250.31.57.anon.1",
           "19.250.31.89.anon.1", "19.28.122.142.anon.1", "19.28.122.183.anon.1", "19.28.122.74.anon.1",
           "19.28.122.75.anon.1", "19.28.122.86.anon.1", "19.28.123.25.anon.1", "19.28.123.50.anon.1",
           "19.28.123.65.anon.1", "19.28.123.94.anon.1", "19.21.109.246.anon.1"]

    def get_sample_rate(f, expected_sample=1000000):
        stdout = subprocess.run("wc -l " + f, shell=True, check=True, stdout=subprocess.PIPE)
        # stdout = subprocess.run("cat {}|cut -f 6|sort|uniq|wc -l".format(f), shell=True, check=True, stdout=subprocess.PIPE)
        return int(stdout.stdout.decode().split()[0]) // expected_sample

    for f in dat:
        # sample_rate = get_sample_rate(f)
        print("{} sample {}".format(f, sample_rate))
        with open("DATA_DIR/" + f, "r") as ifile:
            with open("DATA_DIR/" + f + ".sample." + str(sample_rate), "w") as ofile:
                for line in ifile:
                    if not line.strip():
                        continue
                    # 1499275266.441  185.232.99.68   17      1296    263     853309c7e147d89cef9de38c5840bd4512ead797a17ca8c9fb2660073be35da6        206     0       1
                    line_split = line.split("\t")
                    if hash(line_split[5]) % sample_rate < 1:
                        if len(line_split) < 6:
                            raise RuntimeError("{} {}".format(len(line_split), line_split))
                        ofile.write(line)


def convert_trace(dat="DATA_DIR/185.232.99.68.anon.1"):
    reader = CsvReader(dat, init_params=AKAMAI_CSV3)
    # reader = VscsiReader("../../data/trace.vscsi")
    last_access_time = {}
    access_time_list_dict = defaultdict(list)
    for n, r in enumerate(reader):
        if r in last_access_time:
            access_time_list_dict[r].append(n - last_access_time[r])
        last_access_time[r] = n

    with open(reader.file_loc+".ts_df_list.pickle", "wb") as ofile:
        pickle.dump(access_time_list_dict, ofile, protocol=4)
    return access_time_list_dict

def get_fit_params(obj, ts_df_list, use_log, log_base, func_name):
    t = time.time()
    ts_df_cnt_list = [0] * int(math.log(max(ts_df_list), log_base) + 2)
    for ts_df in ts_df_list:
        ts_df_cnt_list[int(math.log(ts_df, log_base)) + 1] += 1
    if not use_log:
        xdata = [log_base ** i for i in range(len(ts_df_cnt_list))]
    else:
        # ts_df_cnt_list = [0] * (max(ts_df_list) + 1)
        # for ts_df in ts_df_list:
        #     ts_df_cnt_list[ts_df] += 1
        xdata = [i for i in range(len(ts_df_cnt_list))]
    # print("A {} s".format(time.time() - t))

    # CDF
    for i in range(1, len(ts_df_cnt_list)):
        ts_df_cnt_list[i] += ts_df_cnt_list[i - 1]
    # print("B {} s".format(time.time() - t))

    # normalization
    for i in range(len(ts_df_cnt_list)):
        ts_df_cnt_list[i] /= ts_df_cnt_list[-1]
    # print("C {} s".format(time.time() - t))

    # bridge the beginning
    pos = 0
    for i in range(len(ts_df_cnt_list)):
        if ts_df_cnt_list[i] != 0:
            pos = i
            break
    for i in range(0, pos):
        ts_df_cnt_list[i] = ts_df_cnt_list[pos]
    # print("D {} s".format(time.time() - t))

    try:
        popt, sigmoid_func = sigmoid_fit(xdata, ts_df_cnt_list, func_name)
        # print("E {} s".format(time.time() - t))
        return obj, popt
    except Exception as e:
        print(e)
        print("{} failed to fit {} {}".format(obj, ts_df_cnt_list, xdata))
        return None

def ASig_offline(dat=DATA_DIR + "185.232.99.68.anon.1", func_name="arctan",
                 freq_upbound=2000000, freq_lower_bound=8, use_log=False, log_base=1.2):
    if not os.path.exists(dat+".ts_df_list.pickle"):
        access_time_list_dict = convert_trace(dat)
    else:
        with open(dat+".ts_df_list.pickle", "rb") as ifile:
            access_time_list_dict = pickle.load(ifile)

    fit_params = {}
    futures_dict = {}
    n = 0

    # for obj, ts_df_list in access_time_list_dict.items():
    #     t = time.time()
    #     get_fit_params(obj, ts_df_list, use_log, log_base, func_name)
    #     print("finish one with len {} in {} s".format(len(access_time_list_dict[obj]), time.time() - t))

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ppe:
        for obj, ts_df_list in access_time_list_dict.items():
            if len(ts_df_list) > freq_upbound or len(ts_df_list) < freq_lower_bound:
                n += 1
                if n%2000 == 0:
                    print("ignore {}".format(n))
                continue
            futures_dict[ppe.submit(get_fit_params, obj, ts_df_list, use_log, log_base, func_name)] = obj
        for futures in as_completed(futures_dict):
            obj = futures_dict[futures]
            result = futures.result()
            if result is None:
                n += 1
                continue
            assert obj == result[0]
            fit_params[obj] = tuple(result[1])

            if len(fit_params) + n % 2000 == 0:
                print("finish one with len {}, {}/{}".format(len(access_time_list_dict[obj]), len(fit_params)+n, len(access_time_list_dict)))

            # if int((len(fit_params)+n)/len(access_time_list_dict) * 100) > last_print_percent:
            #     print(str((len(fit_params)+n)/len(access_time_list_dict) * 100)[:5] + "%")
            #     last_print_percent = int((len(fit_params)+n)/len(access_time_list_dict) * 100)


    # print(fit_params)
    with open(dat+".fit.{}.{}.{}".format(func_name, use_log, log_base), "w") as ofile:
        json.dump(fit_params, ofile)


def verify_fit_params(dat=DATA_DIR + "185.232.99.68.anon.1", func_name="arctan",
                 use_log=True, log_base=1.2):

    if not os.path.exists(dat+".ts_df_list.pickle"):
        access_time_list_dict = convert_trace(dat)
    else:
        with open(dat+".ts_df_list.pickle", "rb") as ifile:
            access_time_list_dict = pickle.load(ifile)
    with open(dat+".fit.{}.{}.{}".format(func_name, use_log, log_base), "r") as ifile:
        fit_params = json.load(ifile)

    all_y_list = []
    non_fit = 0
    for obj, ts_df_list in access_time_list_dict.items():
        if obj not in fit_params:
            # print("cannot find {}".format(obj))
            non_fit += 1
            continue

        params = fit_params[obj]
        y_list = []
        for ts_df in ts_df_list:
            if use_log:
                ts_df = int(math.log(ts_df, log_base) )+ 1


            if func_name == "arctan":
                y = arctan(ts_df, *params)
            elif func_name == "arctan3":
                y = arctan3(ts_df, *params)
            elif func_name == "tanh2":
                y = tanh2(ts_df, *params)
            else:
                raise RuntimeError("unknown function")

            y_list.append(y)
            if y == 1.0:
                print(ts_df_list, params)

        print("avg {} min {} max {} len {}".format(sum(y_list)/len(y_list), min(y_list), max(y_list), len(y_list)))
        all_y_list.extend(y_list)
    print("all avg {} min {} max {} len {}".format(sum(all_y_list) / len(all_y_list), min(all_y_list), max(all_y_list), len(all_y_list)))


if __name__ == "__main__":
    # ASig_offline(dat="../../data/trace.vscsi")
    # sample_trace(20)
    # sys.exit(1)
    ts_df_list = [31, 28, 28, 32, 37, 16, 13, 14, 43, 15, 23, 20, 47, 20, 46, 8, 61, 15, 34, 16, 24, 98, 23, 48, 448, 237, 208, 236,
     543, 611, 300, 253, 210, 253, 242, 220, 182, 300, 221, 232, 278, 313, 230, 265, 244, 219, 238, 255, 269, 241, 270,
     896, 475, 344, 277, 249, 207, 270, 287, 248, 303, 258, 348, 237, 256, 241, 268, 270, 321, 257, 245, 305, 571, 483,
     218, 286, 180, 227, 239, 247, 231, 213, 243, 290, 245, 246, 202, 242, 253, 226, 251, 284, 233, 252, 208, 619, 536,
     256, 271, 248, 294, 226, 268, 317, 310, 310, 349, 286, 269, 225, 223, 272, 242, 243, 223, 240, 216, 391, 549, 573,
     399, 199, 199, 205, 238, 228, 219, 237, 310, 200, 259, 259, 232, 228, 248, 241, 242, 233, 788, 243, 239, 258, 293,
     274, 291, 243, 287, 248, 299, 276, 222, 222, 252, 255, 330, 282, 295, 268, 295, 1202, 286, 317, 382, 265, 296, 306,
     237, 230, 216, 244, 195, 124, 26, 36, 39, 20, 191, 265, 240, 234, 230, 242, 522, 394, 335, 207, 358, 310, 328, 299,
     307, 312, 253, 254, 281, 275, 803, 264, 312, 278, 259, 237, 267, 256, 281, 255, 1051, 528, 273, 275, 256, 249, 278,
     282, 283, 275, 239, 259, 224, 242, 247, 247, 249, 302, 252, 269, 216, 245, 674, 451, 293, 289, 293, 304, 301, 284,
     279, 222, 263, 235, 226, 262, 286, 254, 316, 283, 300, 257, 279, 224, 207, 390, 568, 578, 255, 154, 218, 247, 287,
     266, 265, 215, 210, 217, 254, 280, 288, 279, 300, 260, 282, 323, 314, 240, 228, 262, 635, 352, 317, 345, 581, 277,
     236, 281, 320, 277, 220, 260, 369, 327, 461, 286, 287, 370, 306, 299, 264, 256, 688, 242, 268, 248, 287, 309, 279,
     333, 392, 308, 290, 305, 298, 295, 350, 446, 298, 256, 288, 289, 296, 285, 255, 725, 388, 279, 291, 313, 337, 313,
     242, 298, 250, 267, 304, 312, 317, 294, 312, 259, 390, 330, 294, 289, 304, 743, 625, 477, 254, 293, 235, 231, 193,
     290, 280, 269, 340, 273, 227, 280, 253, 297, 278, 288, 258, 245, 291, 753, 500, 274, 288, 269, 291, 308, 268, 280,
     301, 337, 264, 311, 240, 249, 257, 252, 299, 278, 232, 236, 191, 271, 286, 546, 373, 461, 393, 256, 168, 300, 242,
     223, 199, 223, 220, 226, 224, 241, 256, 267, 250, 235, 262, 201, 194, 199, 186, 455, 300, 433, 192, 197, 204, 211,
     219, 221, 222, 234, 226, 269, 209, 319, 289, 226, 238, 224, 194, 195, 211, 237, 196, 277, 526, 190, 226, 239, 220,
     243, 275, 206, 259, 210, 230, 234, 267, 243, 218, 322, 232, 310, 284, 227, 230, 197, 222, 217, 222, 272, 337, 388,
     357, 529, 247, 216, 379, 242, 228, 241, 216, 251, 218, 240, 207, 221, 223, 226, 231, 270, 246, 255, 217, 655, 397,
     222, 197, 186, 230, 264, 233, 284, 311, 339, 254, 232, 202, 239, 216, 244, 273, 225, 261, 227, 200, 284, 209, 348,
     416, 455, 504, 238, 234, 208, 248, 237, 260, 321, 257, 259, 282, 291, 334, 261, 274, 222, 265, 235, 260, 240, 298,
     267, 385, 294, 317, 324, 242, 273, 281, 193, 238, 234, 185, 245, 218, 236, 256, 200, 240, 271, 299, 240, 225, 225,
     353, 733, 272, 266, 267, 276, 276, 228, 217, 264, 266, 250, 226, 268, 219, 238, 235, 209, 272, 318, 265, 246, 242,
     250, 653, 598, 226, 209, 344, 291, 247, 246, 220, 251, 261, 263, 235, 233, 360, 296, 257, 189, 215, 197, 243, 309,
     200, 306, 463, 491, 229, 341, 253, 242, 255, 257, 286, 217, 247, 212, 251, 191, 270, 200, 227, 252, 230, 194, 257,
     238, 273, 371, 574, 270, 252, 311, 267, 317, 252, 247, 214, 212, 199, 240, 226, 216, 243, 233, 245, 241, 232, 233,
     210, 185, 203, 224, 356, 446, 281, 417, 247, 238, 217, 182, 170, 164, 208, 249, 242, 248, 250, 217, 233, 229, 248,
     203, 202, 216, 261, 193, 552, 199, 220, 261, 259, 310, 246, 318, 258, 310, 256, 259, 212, 206, 234, 270, 254, 275,
     311, 235, 297, 302, 262, 261, 235, 277, 277, 239, 248, 281, 443, 292, 270, 248, 246, 240, 283, 311, 307, 251, 450,
     585, 297, 229, 225, 302, 249, 233, 276, 279, 250, 249, 276, 226, 234, 247, 232, 243, 188, 224, 247, 250, 222, 189,
     240, 291, 258, 227, 229, 218, 237, 241, 287, 250, 234, 238, 231, 272, 219, 739, 383, 301, 226, 231, 210, 252, 280,
     270, 272, 232, 184, 234, 209, 249, 237, 211, 234, 288, 262, 232, 250, 280, 266, 521, 390, 352, 476, 267, 265, 233,
     228, 222, 235, 210, 191, 218, 246, 202, 240, 229, 233, 216, 158, 210, 204, 223, 203, 377, 396, 343, 330, 200, 188,
     168, 215, 258, 195, 217, 246, 253, 193, 217, 198, 240, 242, 236, 197, 260, 203, 247, 255, 436, 329, 385, 324, 202,
     188, 242, 204, 408698, 193, 208, 224, 237, 219, 208, 256, 214, 224, 280, 226, 229, 220, 233, 280, 238, 234, 230,
     213, 253, 237, 200, 253, 268, 285, 207, 203, 278, 294, 233, 231, 235, 228, 230, 225, 183, 229, 257, 239, 211, 220,
     197, 216, 209, 237, 217, 232, 236, 294, 285, 274, 255, 188, 159, 230, 296, 206, 199, 297, 221, 219, 215, 248, 236,
     228, 237, 236, 242, 248, 239, 238, 236, 225, 240, 265, 187, 239, 271, 279, 290, 487, 300, 293, 290, 277, 265, 309,
     274, 303, 270, 304, 338, 278, 287, 331, 268, 257, 246, 331, 312, 369, 285, 358, 331, 296, 286, 250, 267, 268, 248,
     278, 290, 304, 275, 246, 264, 311, 308, 284, 265, 245, 345, 299, 314, 351, 388, 274, 259, 236, 242, 224, 275, 235,
     226, 293, 220, 256, 224, 267, 272, 228, 246, 256, 324, 233, 275, 295, 308, 288, 287, 282, 267, 268, 307, 314, 308,
     327, 340, 305, 330, 280, 286, 269, 310, 301, 283, 289, 310, 276, 291, 296, 307, 290, 333, 298, 310, 315, 321, 281,
     307, 286, 301, 304, 275, 279, 339, 289, 350, 343, 331, 334, 372, 366, 305, 325, 272, 273, 269, 251, 260, 304, 290,
     290, 235, 309, 310, 247, 310, 264, 272, 263, 293, 328, 272, 284, 282, 305, 304, 312, 286, 295, 336, 245, 385, 303,
     289, 249, 253, 238, 284, 290, 264, 256, 231, 201, 239, 245, 241, 223, 209, 258, 267, 250, 254, 246, 240, 281, 278,
     227, 218, 244, 238, 227, 235, 246, 280, 224, 239, 211, 221, 211, 202, 214, 238, 289, 217, 242, 263, 210, 216, 228,
     247, 220, 272, 273, 235, 261, 296, 326, 312, 244, 263, 274, 322, 359, 287, 291, 263, 203, 231, 262, 282, 315, 275,
     232, 286, 253, 359, 351, 287, 303, 236, 234, 264, 262, 257, 251, 241, 258, 311, 242, 361, 266, 290, 302, 291, 326,
     341, 327, 319, 316, 243, 259, 280, 290, 255, 249, 282, 239, 243, 342, 242, 217, 220, 260, 246, 266, 218, 288, 272,
     287, 248, 265, 277, 265, 372, 320, 300, 258, 254, 278, 255, 302, 269, 255, 265, 258, 231, 247, 286, 266, 199, 277,
     280, 267, 249, 269, 260, 287, 226, 307, 272, 277, 277, 229, 263, 265, 280, 248, 229, 299, 271, 280, 311, 266, 254,
     235, 258, 298, 330, 283, 272, 298, 230, 262, 279, 291, 242, 277, 233, 259, 276, 298, 315, 256, 283, 246, 224, 328,
     313, 296, 310, 301, 281, 447, 297, 259, 226, 256, 234, 234, 242, 264, 378, 534, 364, 324, 374, 270, 310, 243, 278,
     256, 215, 254, 219, 224, 248, 233, 318, 277, 272, 264, 242, 307, 242, 249, 257, 253, 275, 267, 324, 336, 269, 260,
     222, 233, 304, 293, 289, 253, 272, 234, 289, 250, 260, 299, 242, 311, 286, 262, 300, 342, 293, 267, 296, 280, 229,
     276, 291, 245, 231, 268, 279, 209, 254, 233, 257, 287, 295, 248, 288, 273, 276, 287, 296, 258, 291, 291, 287, 292,
     266, 274, 246, 230, 279, 269, 286, 290, 265, 243, 271, 242, 280, 331, 298, 357, 267, 276, 300, 225, 235, 216, 289,
     252, 297, 464, 267, 331, 337, 371, 369, 294, 279, 285, 262, 253, 244, 318, 305, 367, 341, 323, 268, 267, 266, 264,
     246, 282, 254, 295, 246, 266, 270, 266, 236, 267, 279, 331, 267, 247, 262, 305, 280, 229, 257, 279, 208, 204, 227,
     227, 270, 275, 347, 273, 263, 244, 258, 292, 287, 262, 252, 238, 294, 325, 292, 277, 243, 179, 233, 278, 305, 234,
     248, 285, 329, 232, 253, 217, 247, 213, 267, 242, 202, 195, 249, 241, 300, 311, 287, 298, 257, 265, 332, 251, 264,
     274, 249, 236, 224, 198, 223, 245, 219, 255, 237, 243, 275, 321, 310, 271, 356, 340, 263, 274, 269, 302, 229, 270,
     234, 299, 344, 261, 270, 209, 323, 217, 277, 240, 261, 291, 218, 198, 225, 240, 225, 293, 234, 298, 273, 242, 236,
     222, 279, 215, 240, 234, 262, 256, 291, 305, 299, 242, 284, 305, 307, 270, 281, 246, 362, 354, 263, 285, 346, 280,
     246, 252, 280, 285, 269, 275, 269, 306, 286, 294, 224, 237, 278, 305, 352, 292, 277, 284, 255, 230, 259, 288, 236,
     222, 241, 269, 291, 315, 377, 311, 362, 286, 282, 414, 477, 350, 365, 261, 345, 299, 264, 303, 328, 280, 330, 329,
     354, 272, 448, 386, 267, 289, 292, 277, 289, 291, 289, 389, 312, 338, 289, 338, 313, 296, 292, 373, 303, 311, 246,
     261, 266, 305, 318, 302, 305, 317, 308, 300, 302, 363, 389, 371, 370, 336, 283, 297, 339, 348, 342, 279, 293, 264,
     260, 303, 351, 466, 305, 349, 309, 381, 385, 347, 325, 276, 309, 355, 387, 351, 479, 468, 392, 346, 340, 300, 297,
     329, 253, 321, 319, 303, 291, 334, 352, 299, 256, 295, 437, 406, 323, 342, 422, 376, 370, 416, 310, 285, 388, 300,
     262, 245, 280, 228, 231, 240, 211, 225, 207, 208, 272, 240, 292, 307, 220, 215, 207, 227, 178, 210, 221, 191, 211,
     220, 244, 220, 243, 256, 215, 203, 205, 254, 237, 179, 197, 173, 181, 192, 175, 254, 160, 251, 247, 180, 179, 215,
     219, 287, 200, 205, 193, 196, 190, 210, 222, 218, 181, 249, 208, 200, 223, 216, 224, 173, 181, 189, 240, 252,
     1390500, 17590877]
    # [0.7991367064831277, -24.440559985661107]
    # [0.09169168865749719, -5.263606348838008]

    # popt = get_fit_params("test", ts_df_list, True, 1.2, "tanh2")
    # print(popt)

    verify_fit_params(func_name="arctan3")

    # ASig_offline()
    # ASig_offline(func_name="arctan3")
    # ASig_offline(func_name="tanh2")