# coding=utf-8
"""
this is a module facilitates the printing and screen logging
multithreading printing is supported, so message won't interleave.

define print_level before importing this module
if you want to control the printing verbosity
print_level     verbosity
1               NOT USED
2               DEBUG
3               INFO
4               WARNING
5               ERROR
"""


import sys
import time
import traceback
from threading import Lock

COLOR_RED           = '\033[91m'
COLOR_GREEN         = '\033[92m'
COLOR_YELLOW        = '\033[93m'
COLOR_LIGHT_PURPLE  = '\033[94m'
COLOR_PURPLE        = '\033[95m'
COLOR_BLUE          = '\033[34m'
COLOR_END           = '\033[0m'

fcolor_dict = {'red': COLOR_RED, 'green': COLOR_GREEN, 'yellow': COLOR_YELLOW,
                'blue': COLOR_BLUE, 'purple': COLOR_PURPLE}

bcolor_dict = {'red': '\033[41m', 'green': '\033[42m', 'yellow': '\033[43m',
                'blue': '\033[44m', 'purple': '\033[45m',
               'cyan': '\033[46m', 'white': '\033[47m', 'end': '\033[49m'}


if "print_level" not in globals():
    # default level: INFO
    print_level = 3
printing_lock = Lock()

def colorful_print(color, s):
    """ print a message with color
    """

    printing_lock.acquire()
    print('{}{}{}'.format(fcolor_dict[color.lower()], s, COLOR_END))
    printing_lock.release()


def colorful_print_with_background(bcolor, fcolor, s):
    """print a message with color and background
    """

    printing_lock.acquire()
    print('{}{}{}{}'.format(bcolor_dict[bcolor.lower()],
                            fcolor_dict[fcolor.lower()], s, COLOR_END))
    printing_lock.release()



def print_list(l, num_per_line=20):
    """ elegantly print a list
    """

    printing_lock.acquire()
    counter = 0
    for i in l:
        print("{}".format(i), end="\t")
        counter += 1
        if counter%num_per_line==0:
            print("")
            counter = 0
    printing_lock.release()


def DEBUG(s):
    """
    level 2
    """
    if print_level >= 2:
        printing_lock.acquire()
        print('[DEBUG]: {}: {}{}{}'.format(time.strftime("%H:%M:%S", time.localtime(time.time())),
                                           COLOR_LIGHT_PURPLE, s, COLOR_END))
        printing_lock.release()

def INFO(s):
    """
    level 3
    """
    if print_level >= 3:
        printing_lock.acquire()
        print('[INFO]: {}: {}{}{}'.format(time.strftime("%H:%M:%S", time.localtime(time.time())),
                                          COLOR_YELLOW, s, COLOR_END))
        printing_lock.release()

def WARNING(s):
    """
    level 4
    """
    if print_level >= 4:
        printing_lock.acquire()
        print('[WARNING]: {}: {}{}{}'.format(time.strftime("%H:%M:%S", time.localtime(time.time())),
                                             COLOR_PURPLE, s, COLOR_END, file=sys.stderr))
        printing_lock.release()

def ERROR(s):
    """
    level 5
    """
    if print_level >= 5:
        printing_lock.acquire()
        print('[ERROR]: {}: {}{}{}'.format(time.strftime("%H:%M:%S", time.localtime(time.time())),
                                           COLOR_RED, s, COLOR_END), file=sys.stderr)
        printing_lock.release()




if __name__ == "__main__":
    colorful_print("purple", "T")
    colorful_print_with_background("yellow", 'blue', 'text')
    print('normal')
    DEBUG("debug")

