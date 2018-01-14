# coding=utf-8


"""
    this module is used for conveniently print running time

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/10

"""

import os
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PyMimircache.utils.printing import *

class MyTimer:
    def __init__(self):
        self.begin = time.time()
        self.last_tick_time = time.time()


    def tick(self, msg=None):
        INFO("{}time since last tick {} seconds, total time {} seconds".format(
            "" if msg is None else "{} ".format(msg) ,
            time.time() - self.last_tick_time, time.time() - self.begin
        ))
        self.last_tick_time = time.time()


