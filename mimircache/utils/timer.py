# coding=utf-8


"""
this module is used for conveniently print running time

"""

import os
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mimircache.utils.printing import *

class myTimer:
    def __init__(self):
        self.begin = time.time()
        self.last_tick_time = time.time()


    def tick(self, msg=None):
        INFO("{} time since last tick {} seconds, total time {} seconds".format(
            "" if msg is None else msg,
            time.time() - self.last_tick_time, time.time() - self.begin
        ))
        self.last_tick_time = time.time()


