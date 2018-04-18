# coding=utf-8


import os, sys, pickle
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
from PyMimircache import *


DIR = "/home/jason/ALL_DATA/Akamai/"
DIR_DS_SPLIT = "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/"

def _generate_global_hashtable(dat):
    """
    this function takes original data and generate one to one mapping from 
    key in original data to new key 
    :param dat: 
    :return: the number of unique keys 
    """
    d = {}
    count = 0
    with open("{}/{}".format(DIR, dat)) as ifile:
        for line in ifile:
            key = line.split("\t")[4]
            if key not in d:
                d[key] = len(d) + 1
            count += 1
            if count % 10000000 == 0:
                print(count)
    with open("{}/global.hash".format(DIR), 'wb') as ofile:
        pickle.dump(d, ofile)
    return len(d)


def get_global_hashtable(dat):
    """
    check whether global hashtable exists or not, if yes, load it and return
    if not, generate it, save it and return 
    :param dat: 
    :return: 
    """
    if os.path.exists("/run/shm/global.hash"):
        path = "/run/shm/global.hash"
    elif os.path.exists("{}/global.hash".format(DIR)):
        path = "{}/global.hash".format(DIR)
    else:
        path = None

    if path is None:
        return _generate_global_hashtable(dat)
    else:
        with open(path, 'rb') as ifile:
            d = pickle.load(ifile)
        print("hashtable size {}".format(len(d)))
        return d



def convert_compress(reader, global_hash, ts_bit=-1):
    fmt = "<"
    n_requests = reader.get_num_of_req()
    n_uni_req = len(global_hash)

    if n_uni_req > 2 ** 32 - 2:            # use 2 because mapped the label begins with 1 not 0
        label_bits = 64
    elif n_uni_req > 2 ** 16 - 2:
        label_bits = 32
    elif n_uni_req > 2 ** 8 - 2:
        label_bits = 16
    else:
        raise RuntimeError("number of unique keys {}".format(n_uni_req))

    for n_bit in [label_bits, ts_bit]:
        if n_bit != -1:
            if n_bit == 64:
                fmt += "Q"
            elif n_bit == 32:
                fmt += "L"
            elif n_bit == 16:
                fmt += "H"
            else:
                print("cannot recognize given timestamps bits {}".format(ts_bit), file=sys.stderr)

    line_count = 0      # used for output progress
    binary_dir = os.path.dirname(reader.file_loc.replace("csv", "binary"))
    if not os.path.exists(binary_dir):
        os.makedirs(binary_dir)

    if os.path.exists(reader.file_loc.replace("csv", "binary")):
        return

    writer = TraceBinaryWriter(reader.file_loc.replace("csv", "binary"), fmt=fmt)
    print(fmt)

    if ts_bit != -1:
        e = reader.read_time_req()
        first_ts = int(e[0])

        while e:
            t, r = e
            t = int(t)
            writer.write((t, global_hash[r]))
            e = reader.read_time_req()

            line_count +=1
            if line_count % 10000 == 0:
                print("\r{:.2f}%".format(line_count * 100 / n_requests), end='')

    else:
        for r in reader:
            writer.write((global_hash[r], ))

            line_count +=1
            if line_count % 10000 == 0:
                print("\r{:.2f}%".format(line_count * 100 / n_requests), end='')

    writer.close()



if __name__ =="__main__":
    DAT = "201610.all.sort.clean"

    global_hash = get_global_hashtable(DAT)
    for folder in os.listdir(DIR_DS_SPLIT):
        for f in os.listdir("{}/{}".format(DIR_DS_SPLIT, folder)):
            if not os.path.isfile("{}/{}/{}".format(DIR_DS_SPLIT, folder, f)):
                continue
            r = CsvReader("{}/{}/{}".format(DIR_DS_SPLIT, folder, f),
                          init_params={"real_time_column": 1, "label_column": 5, "delimiter": "\t"})
            convert_compress(reader=r, global_hash=global_hash, ts_bit=32)
            print("{}/{}".format(folder, f))
