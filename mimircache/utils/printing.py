# coding=utf-8
import sys, time, traceback

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_LIGHT_PURPLE = '\033[94m'
COLOR_PURPLE = '\033[95m'
COLOR_BLUE = '\033[34m'
COLOR_END = '\033[0m'

fcolor_dict = {'red': COLOR_RED, 'green': COLOR_GREEN, 'yellow': COLOR_YELLOW, 'blue': COLOR_BLUE, 'purple': COLOR_PURPLE}

bcolor_dict = {'red': '\033[41m', 'green': '\033[42m', 'yellow': '\033[43m', 'blue': '\033[44m', 'purple': '\033[45m',
               'cyan': '\033[46m', 'white': '\033[47m', 'end': '\033[49m'}


def colorfulPrint(color, s):
    print('{}{}{}'.format(fcolor_dict[color.lower()], s, COLOR_END))


def colorfulPrintWithBackground(bcolor, fcolor, s):
    print('{}{}{}{}'.format(bcolor_dict[bcolor.lower()], fcolor_dict[fcolor.lower()], s, COLOR_END))


# drop it
def DEBUG_MSG(s):
    print("{}: {}: {}".format(sys._getframe().f_code.co_name, sys._getframe().f_lineno, s))


def printList(l, num_in_one_line=20):
    counter = 0
    for i in l:
        print("{}".format(i), end="\t")
        counter += 1
        if counter%num_in_one_line==0:
            print("")
            counter = 0

def DEBUG(s):
    print('[DEBUG]: {}: {}{}{}'.format(time.time(), COLOR_LIGHT_PURPLE, s, COLOR_END))

def INFO(s):
    print('[INFO]: {}: {}{}{}'.format(time.time(), COLOR_YELLOW, s, COLOR_END))

def WARNING(s):
    print('[WARNING]: {}: {}{}{}'.format(time.time(), COLOR_PURPLE, s, COLOR_END, file=sys.stderr))

def ERROR(s):
    print('[ERROR]: {}: {}{}{}'.format(time.time(), COLOR_RED, s, COLOR_END), file=sys.stderr)




if __name__ == "__main__":
    colorfulPrint("purple", "T")
    colorfulPrintWithBackground("yellow", 'blue', 'text')
    print('normal')
    DEBUG("debug")
