import os

# global c_available_cache
c_available_cache = []
BASE_DIR = os.path.dirname(__file__)


def init():
    init_C_available_cache()


def init_C_available_cache():
    import configparser
    config = configparser.ConfigParser()
    # print(BASE_DIR + '/conf')
    config.read(BASE_DIR + '/conf.py')
    if 'C_available_cache' in config.sections():
        c_available_cache.extend(config['C_available_cache'])
    else:
        raise RuntimeWarning("cannot find any cache module in C")
