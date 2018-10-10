# coding=utf-8
"""
ATTENTION: this module should not be used on splitted data!!! OTHERWISE, SAME KEY in different files will become different
this module converts given trace to binary,
meanwhile replace all the label with increasing number
the saved binary has the form of ts, label, where ts is optional
"""

from PyMimircache import *
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
import os, csv, sys, struct


sys.path.append("../")
sys.path.append("./")
from PyMimircache import *
from PyMimircache.bin.conf import *

################################## Global variable and initialization ###################################

TRACE_TYPE = "Akamai"

TRACE_DIR, NUM_OF_THREADS = initConf(TRACE_TYPE, trace_format=None)




def convert_compress(reader, label_bits=-1, ts_bit=-1):
    fmt = "<"
    n_requests = reader.get_num_of_req()
    if label_bits == -1:
        if n_requests > 2 ** 32 - 2:            # use 2 because mapped the label begins with 1 not 0
            label_bits = 64
        elif n_requests > 2 ** 16 - 2:
            label_bits = 32
        elif n_requests > 2 ** 8 - 2:
            label_bits = 16

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

    d = {}      # label to mapped label
    counter = 1
    line_count = 0      # used for output progress
    writer = TraceBinaryWriter(reader.file_loc + ".mapped.bin." + fmt[1:], fmt=fmt)
    print(fmt)

    if ts_bit != -1:
        e = reader.read_time_req()
        first_ts = int(e[0])

        while e:
            t, r = e
            t = int(t)
            if r in d:
                writer.write((d[r], t-first_ts))
            else:
                writer.write((counter, t-first_ts))
                d[r] = counter
                counter += 1
            e = reader.read_time_req()

            line_count +=1
            if line_count % 10000 == 0:
                print("\r{:.2f}%".format(line_count * 100 / n_requests), end='')

    else:
        for r in reader:
            if r in d:
                writer.write((d[r], ))
            else:
                writer.write((counter, ))
                d[r] = counter
                counter += 1

            line_count +=1
            if line_count % 10000 == 0:
                print("\r{:.2f}%".format(line_count * 100 / n_requests), end='')



    writer.close()




if __name__ == "__main__":
    # reader = vscsiReader("../../data/trace.vscsi")
    reader = CsvReader("{}/{}".format(TRACE_DIR, "20161001.sort"), init_params=AKAMAI_CSV)
    convert_compress(reader, label_bits=-1, ts_bit=32)
    # convert_compress(reader, label_bits=64, ts_bit=32)