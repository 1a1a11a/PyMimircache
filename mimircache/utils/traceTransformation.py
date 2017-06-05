# coding=utf-8

import os
from mimircache import vscsiReader
from mimircache.cacheReader.binaryReader import binaryReader
from mimircache.cacheReader.binaryWriter import traceBinaryWriter



def vReaderToPReader(file_path, output_path):
    vReader = vscsiReader(file_path)
    with open(output_path, 'w') as ofile:
        for req in vReader:
            ofile.write("{}\n".format(req))

def vReaderTocsvReader(file_pth, output_path):
    pass


def splitTrace(reader, n, output_folder, prefix=""):
    """
    given a reader, split the content into n small files
    :param reader:
    :param n:
    :param output_folder:
    :param prefix:
    :return:
    """
    total_num = reader.get_num_total_req()
    num_each_file = total_num // n
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ofiles = []
    for i in range(n):
        ofiles.append(open("{}/{}".format(output_folder, prefix+str(i)), 'w'))
    file_p = -1
    for i, e in enumerate(reader):
        if i % num_each_file == 0 and file_p!=len(ofiles)-1:
            file_p += 1
        ofiles[file_p].write("{}\n".format(e))

    for i in range(n):
        ofiles[i].close()


def trace_mixer(reader1, reader2, mix_mode, output="mixTrace.csv", *args, **kwargs):
    """
    mix two traces into one,
    :param output:
    :param reader1:
    :param reader2:
    :param mix_mode: "real_time", "round_robin"
    :return:
    """
    ofile = open(output, 'w')
    if mix_mode == "round_robin":
        if "round_robin_n" in kwargs:
            round_robin_n = kwargs["round_robin_n"]
        else:
            round_robin_n = 1
        begin_flag = True
        r1 = None
        r2 = None

        while begin_flag or r1 or r2:
            if begin_flag or r1:
                for i in range(round_robin_n):
                    r1 = reader1.read_one_element()
                    if r1:
                        ofile.write("{},{}\n".format('A', 'A' + str(r1)))
                    else:
                        break
            if begin_flag or r2:
                for i in range(round_robin_n):
                    r2 = reader2.read_one_element()
                    if r2:
                        ofile.write("{},{}\n".format('B', 'B' + str(r2)))
                    else:
                        break

            begin_flag = False



    elif mix_mode == "real_time":
        r1 = reader1.read_time_request()
        r2 = reader2.read_time_request()
        init_t1 = r1[0]
        init_t2 = r2[0]
        while r1 or r2:
            if r1[0] - init_t1 <= r2[0] - init_t2:
                ofile.write("{},{},{}\n".format('A', r1[0]-init_t1, 'A' + str(r1[1])))
                r1 = reader1.read_time_request()
            else:
                ofile.write("{},{},{}\n".format('B', r2[0]-init_t2, 'B' + str(r2[1])))
                r2 = reader2.read_time_request()
            if not r1:
                while r2:
                    ofile.write("{},{},{}\n".format('B', r2[0]-init_t2, 'B' + str(r2[1])))
                    r2 = reader2.read_time_request()
            if not r2:
                while r1:
                    ofile.write("{},{},{}\n".format('A', r1[0]-init_t1, 'A' + str(r1[1])))
                    r1 = reader1.read_time_request()


    else:
        print("do not support given mix mode {}".format(mix_mode), file=sys.stderr)


    ofile.close()


def splitTraceWithTime(dat, timeInterval=24*3600*10**6):
    """
    split trace according to timeInterval, save trace as plainText
    :param dat:
    :param timeInterval:
    :return:
    """
    FOLDER_NAME = "{}".format(dat[dat.rfind('/')+1 : dat.find("_")])
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    reader = vscsiReader(dat)
    t, label = reader.read_time_request()
    init_t = t
    ofile = None
    day = 1
    while True:
        if not ofile:
            ofile = open("{}/{}_day{}.txt".format(FOLDER_NAME, dat[dat.rfind('/')+1 : ], day), 'w')
        ofile.write("{}\n".format(label))
        r = reader.read_time_request()
        if r:
            t, label = r
        else:
            break
        if t - init_t > timeInterval:
            print("day {} finished".format(day))
            init_t = t
            ofile.close()
            ofile = None
            day += 1

    ofile.close()


def extract_part_trace(reader, writer, range):
    """
    extract range from reader and save to writer
    :param reader:
    :param writer:
    :param range:
    :return:
    """
    assert len(range) == 2, "range must be in the form of (min, max)"
    maxLine = range[1]
    if maxLine == -1:
        maxLine = len(reader)

    line_count = 0
    # reader.jumpN(range[0])
    for line in reader:
        line_count += 1
        if line_count-1 < range[0]:     # -1 because we add 1 before checking
            continue
        if line_count-1 >= maxLine:
            break
        writer.write(line)




if __name__ == "__main__":
    # vReaderToPReader(sys.argv[1], sys.argv[2])
    # reader = vscsiReader("../data/traces/w38_vscsi1.vscsitrace")
    # reader1 = vscsiReader("../data/trace.vscsi")
    # reader1 = vscsiReader("../data/traces/w106_vscsi1.vscsitrace")
    # reader2 = vscsiReader("../data/traces/w100_vscsi1.vscsitrace")
    # trace_mixer(reader1, reader2, mix_mode="round_robin", round_robin_n=2)
    # trace_mixer(reader1, reader2, mix_mode="real_time")
    # splitTrace(reader, 2, "w38Split")
    # splitTraceWithTime("/var/dept/scratch/jyan254/cache/mimircache/data/traces/w65_vscsi1.vscsitrace")
    with binaryReader("../data/trace.vscsi", "<3I2H2Q") as reader:
        with traceBinaryWriter("test.vscsi", "<3I2H2Q") as writer:
            extract_part_trace(reader, writer, (1, -1))
    with binaryReader("../utils/test.vscsi", "<3I2H2Q") as r:
        print(len(r))

