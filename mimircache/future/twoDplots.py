# coding=utf-8
"""
this module contains new functions that going to be added to future version of mimircache
"""

from mimircache import *
from mimircache.profiler.twoDPlots import *
from collections import defaultdict

def freq_distr(reader):
    """
    calculate the frequency distribution of requests in the reader,
    will be moved the reader module
    :param reader:
    :param figname:
    :return:
    """
    d = defaultdict(int)
    for r in reader:
        d[r] += 1
    d_freq_distr = defaultdict(int)
    for _, v in d.items():
        d_freq_distr[v] += 1
    reader.reset()
    return d_freq_distr


def freq_distr_2d(reader, figname="freq_distri2d.png"):
    d = freq_distr(reader)
    max_freq = max(d.keys())
    l = [0] * max_freq
    for k,v in d.items():
        l[k-1] = v
    draw2d(l, xlabel="frequency", ylabel="count", logX=True, logY=True, figname=figname)


if __name__ == "__main__":
    reader = vscsiReader("../../data/trace.vscsi")
    reader = vscsiReader("/scratch/jason/traces/w106_vscsi1.vscsitrace")
    freq_distr_2d(reader, "test_freq.pdf")

