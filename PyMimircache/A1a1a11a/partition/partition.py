# coding=utf-8
from PyMimircache import *
from collections import deque, defaultdict
from PyMimircache.utils.printing import printList
import csv
import math
from matplotlib import pyplot as plt

import os, sys, time, pickle
from multiprocessing import Pool, Process, cpu_count
from functools import partial
sys.path.append("../")
sys.path.append("./")
from PyMimircache import *
import PyMimircache.c_heatmap as c_heatmap
from PyMimircache.bin.conf import *
from PyMimircache.A1a1a11a.ML.featureGenerator import *

################################## Global variable and initialization ###################################

TRACE_TYPE = "Akamai"
TRACE_DIR, NUM_OF_THREADS = initConf(TRACE_TYPE, trace_format=None)


TRACE_DIR = "/root/disk2/ALL_DATA/Akamai/binary/"
NUM_OF_BINS = 8



