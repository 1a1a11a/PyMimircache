import sys

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
LIGHT_PURPLE = '\033[94m'
PURPLE = '\033[95m'
BLUE = '\033[34m'
END = '\033[0m'

fcolor_dict = {}
fcolor_dict['red'] = RED
fcolor_dict['green'] = GREEN
fcolor_dict['yellow'] = YELLOW
fcolor_dict['blue'] = BLUE
fcolor_dict['purple'] = PURPLE

bcolor_dict = {}
bcolor_dict['red'] = '\033[41m'
bcolor_dict['green'] = '\033[42m'
bcolor_dict['yellow'] = '\033[43m'
bcolor_dict['blue'] = '\033[44m'
bcolor_dict['purple'] = '\033[45m'
bcolor_dict['cyan'] = '\033[46m'
bcolor_dict['white'] = '\033[47m'
bcolor_dict['end'] = '\033[49m'


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
