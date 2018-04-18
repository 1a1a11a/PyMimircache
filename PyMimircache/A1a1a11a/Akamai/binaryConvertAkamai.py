# coding=utf-8

################################## SETTING and CONF ##################################
"""
format of input Akamai log:
timestamp(s)    ip          DataCenter  Traffic                 Request                                             HTTP hit/miss
1477958399	88.15.159.131	395	        1740	85b301a2c33892b3379928dc1f0113754efaa2a5bc1a8f1d50be63ceb18b99b3	200	1

HTTP status code and cache hit/miss is not stored in the binary data
stored binary file format:
struct entry{
    uint32      ts,     // should be enough for current time
    char[16]    ip,     // each part is padded with 0 to reach 3 digits, then padded with another 0 at end for alignment
    uint16      dataCenter,     // data center ID
    uint16      traffic,        // Traffic ID
    char[64]    label,          // request label
}
corresponding python struct fmt: "<I16s2H64c"
"""

BINARY_FMT = "<I16s2H64s"


# from PyMimircache.cacheReader.csvReader import csvReader
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
import os, csv, sys, struct

def convertToBinary(ifile_loc, ofile_loc):
    assert ifile_loc != ofile_loc, "input data and output data are same, input data will be overwritten"
    structIns = struct.Struct(BINARY_FMT)
    # assert ifile_loc.endswith('csv'), "input file must be csv file"
    with open(ifile_loc) as ifile:
        with open(ofile_loc, 'wb') as ofile:
            # reader = csv.reader(ifile, dialect='\t')
            for line in ifile:
                row = line.split('\t')
                assert len(row) >= 6, "current row does not have enough information: {}".format(row)
                ts = int(float(row[0]))
                ip = bytes('.'.join([ '0'*(3-len(p))+p for p in row[1].split('.') ]), 'ascii') # + '0'
                # print(ip)
                dc = int(row[2])
                traffic = int(row[3])
                label = bytes(row[4], 'ascii')
                status = int(row[5])
                if status == 200:
                    ofile.write(structIns.pack(ts, ip, dc, traffic, label))


def convertToCsv(ifile_loc, ofile_loc):
    # with open(ifile_loc, 'rb') as ifile:
    with BinaryReader(ifile_loc, BINARY_FMT) as reader:
        with open(ofile_loc, 'w') as ofile:
            # writer = csv.writer(ofile)
            for e in reader:
                ofile.write("{}\t{}\t{}\t{}\t{}\n".format(e[0], e[1].decode('ascii')[:-1], e[2], e[3], e[4].decode('ascii')))
                # writer.writerow(e)



if __name__ == "__main__":
    TRACE_DIR = "/home/jason/ALL_DATA/akamai_new_logs"
    for f in os.listdir(TRACE_DIR):
        if "bin" in f:
            continue
        print(f)
        convertToBinary("{}/{}".format(TRACE_DIR, f), "{}/{}".format(TRACE_DIR, f.replace('anon', 'bin')))

    # convertToBinary("/home/A1a1a11a/20161001.sort.e", "/home/A1a1a11a/20161001.bin")
    # convertToCsv("/home/A1a1a11a/20161001.bin", "/home/A1a1a11a/20161001.sort.n")

