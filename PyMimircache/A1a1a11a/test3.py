from PyMimircache import *
import pickle
import math
import struct
from collections import defaultdict


from PyMimircache.utils.timer import MyTimer
from PyMimircache.bin.conf import *
from random import randint

def _f(l):
    for i in range(100):
        l[randint(0, len(l)-1)] = randint(0, 100)

def mytest1(N, s):
    for i in range(N):
        l = [0] * s
        _f(l)
        x = sum(l)

def mytest2(N, s):
    l = [0] * s
    for i in range(N):
        for j in range(len(l)):
            l[j] = 0
        _f(l)
        x = sum(l)

def mytest3():
    br = BinaryReader("../data/trace.vscsi", init_params={"size": 2, "op": 4, "label": 6, "real_time": 7, "fmt": "<3I2H2Q"})
    br = VscsiReader("../data/trace.vscsi")
    # br = CsvReader
    print(br.read_last_req()[br.time_column-1])

if __name__ == "__main__":
    mytest3()
