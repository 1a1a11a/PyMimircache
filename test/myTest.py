# coding=utf-8


import os, sys, time
from PyMimircache import Cachecow


def cachecow_test1(dat):
    c = Cachecow()
    c.vscsi(dat)
    c.characterize("long")




if __name__ == "__main__":
    # print(os.listdir("../data"))
    cachecow_test1("../PyMimircache/data/trace.vscsi")