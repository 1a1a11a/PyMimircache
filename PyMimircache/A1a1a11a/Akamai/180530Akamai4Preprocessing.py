# coding=utf-8

import heapq
from PyMimircache import *
from PyMimircache.bin.conf import *


DAT = [
    "/home/jason/ALL_DATA/akamai4/nyc_66",
    "/home/jason/ALL_DATA/akamai4/nyc_85",
    "/home/jason/ALL_DATA/akamai4/nyc_189",
    "/home/jason/ALL_DATA/akamai4/nyc_1731",
    "/home/jason/ALL_DATA/akamai4/nyc_1966",
    "/home/jason/ALL_DATA/akamai4/lax_924",
    "/home/jason/ALL_DATA/akamai4/lax_1319",
    "/home/jason/ALL_DATA/akamai4/lax_1448",
    "/home/jason/ALL_DATA/akamai4/lax_1831",
    "/home/jason/ALL_DATA/akamai4/lax_1870",
    "/home/jason/ALL_DATA/akamai4/sjc_67",
    "/home/jason/ALL_DATA/akamai4/sjc_1384",
    "/home/jason/ALL_DATA/akamai4/sjc_1980",
]


# > ts, ip, datacenter, customer_id, serial, hashobj, size1, size2, traffic_class


def merge_trace(dat, out):
    reader_buffer = []
    ifiles = []
    for i in dat:
        ifiles.append(open(i))
    ofile = open(out, "w")

    for j in range(len(ifiles)):
        line = ifiles[j].readline()
        t = float(line.split("\t")[0])
        reader_buffer.append((t, j, line))

    heapq.heapify(reader_buffer)

    item = heapq.heappop(reader_buffer)
    while item:
        t, ifileID, line = item
        ofile.write(line)
        line = ifiles[ifileID].readline()
        if line:
            t = float(line.split("\t")[0])
            heapq.heappush(reader_buffer, (t, ifileID, line))
        else:
            pass
        if len(reader_buffer):
            item = heapq.heappop(reader_buffer)
        else:
            break

    for ifile in ifiles:
        ifile.close()
    ofile.close()


if __name__ == "__main__":
    merge_trace(["/home/jason/ALL_DATA/akamai4/nyc_66",
                 "/home/jason/ALL_DATA/akamai4/nyc_85",
                 "/home/jason/ALL_DATA/akamai4/nyc_189",
                 "/home/jason/ALL_DATA/akamai4/nyc_1731",
                 "/home/jason/ALL_DATA/akamai4/nyc_1966"],
                out="/home/jason/ALL_DATA/akamai4/mergedTrace/nyc")

    merge_trace(["/home/jason/ALL_DATA/akamai4/lax_924",
                 "/home/jason/ALL_DATA/akamai4/lax_1319",
                 "/home/jason/ALL_DATA/akamai4/lax_1448",
                 "/home/jason/ALL_DATA/akamai4/lax_1831",
                 "/home/jason/ALL_DATA/akamai4/lax_1870"],
                out="/home/jason/ALL_DATA/akamai4/mergedTrace/lax")

    merge_trace(["/home/jason/ALL_DATA/akamai4/sjc_67",
                 "/home/jason/ALL_DATA/akamai4/sjc_1384",
                 "/home/jason/ALL_DATA/akamai4/sjc_1980"],
                out="/home/jason/ALL_DATA/akamai4/mergedTrace/sjc")


    # splitByType