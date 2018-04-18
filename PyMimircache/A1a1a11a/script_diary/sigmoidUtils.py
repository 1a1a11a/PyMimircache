# coding=utf-8
import os
import math
from collections import defaultdict

from PyMimircache.utils.printing import *
from PyMimircache.bin.conf import *
from PyMimircache.A1a1a11a.myUtils.prepareRun import *

def transform_datafile_to_dist_list(dat, dat_type, dist_type="rd"):

    folder_name = "1106_dist_list"
    output_name = "{}_{}".format(dist_type, os.path.basename(dat))

    if prepare(output_name, os.path.abspath(folder_name), False):
        return "{}/{}".format(folder_name, output_name)

    DEBUG("begin generating dist_list file {}/{}".format(folder_name, output_name))
    reader = get_reader(dat, dat_type)
    dist = None
    if dist_type == "rd":
        dist = LRUProfiler(reader).get_reuse_distance()
    d = defaultdict(list)

    for n, req in enumerate(reader):
        d[req].append(dist[n])


    with open("{}/{}".format(folder_name, output_name), "w") as ofile:
        for req, l in sorted(d.items(), key=lambda x: len(x[1]), reverse=True):
            ofile.write("{}: {}\n".format(req, l))
    return "{}/{}".format(folder_name, output_name)


def transform_dist_list_to_dist_count(dist_list, cdf=True, normalization=True, min_dist=-1, log_base=1.20):
    dist_count_list_size = int(math.floor(math.log(max(dist_list) + 1, log_base)) + 1)
    dist_count_list = [0] * dist_count_list_size
    for dist in dist_list:
        if dist != -1:
            dist = dist + 1
            if min_dist != -1 and dist < min_dist:
                dist = min_dist
            dist_count_list[int(math.floor(math.log(dist, log_base)))] += 1
    if cdf:
        for i in range(1, len(dist_count_list)):
            dist_count_list[i] = dist_count_list[i-1] + dist_count_list[i]
        if normalization:
            for i in range(0, len(dist_count_list)):
                dist_count_list[i] = dist_count_list[i] / dist_count_list[-1]
    else:
        if normalization:
            raise RuntimeError("not implemented")

    # now change all points that are disconnected due to min_dist
    pos = 0
    for i in range(len(dist_count_list)):
        if dist_count_list[i] != 0:
            pos = i
            break
    for i in range(0, pos):
        dist_count_list[i] = dist_count_list[pos]

    return dist_count_list


def add_one_rd_to_dist_list(dist, dist_count_list, weight=1, min_dist=-1, base=1.20):
    if dist != -1:
        dist = dist + 1
        if min_dist != -1 and dist < min_dist:
            dist = min_dist
    ind = int(math.floor(math.log(dist, base)))
    if ind > len(dist_count_list):
        to_append = dist_count_list[-1]
        for i in range(ind - len(dist_count_list)):
            dist_count_list.append(to_append)

    for i in range(ind, len(dist_count_list)):
        dist_count_list[i] += weight

    return dist_count_list