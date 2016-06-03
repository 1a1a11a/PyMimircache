import sys

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
LIGHT_PURPLE = '\033[94m'
PURPLE = '\033[95m'
BLUE = '\033[34m'
END = '\033[0m'

fcolor_dict = {'red': RED, 'green': GREEN, 'yellow': YELLOW, 'blue': BLUE, 'purple': PURPLE}

bcolor_dict = {'red': '\033[41m', 'green': '\033[42m', 'yellow': '\033[43m', 'blue': '\033[44m', 'purple': '\033[45m',
               'cyan': '\033[46m', 'white': '\033[47m', 'end': '\033[49m'}


def colorfulPrint(color, s):
    print('{}{}{}'.format(fcolor_dict[color.lower()], s, END))


def colorfulPrintWithBackground(bcolor, fcolor, s):
    print('{}{}{}{}'.format(bcolor_dict[bcolor.lower()], fcolor_dict[fcolor.lower()], s, END))


# drop it
def debugPrint(s):
    print("{}: {}: {}".format(sys._getframe().f_code.co_name, sys._getframe().f_lineno, s))


if __name__ == "__main__":
    colorfulPrint("purple", "T")
    colorfulPrintWithBackground("yellow", 'blue', 'text')
    print('normal')
    debugPrint("debug")
