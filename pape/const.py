
params = {}
# set mining_period = 100000, prefetch_table = 1000000
MINING_LIST_POINTER = 2

MINING_PERIOD       = 5000
PREFETCH_TABLE      = 30000
PREFETCH_LIST_SIZE  = 2


MINING_PERIOD2       = 5000
PREFETCH_TABLE2      = 120000
PREFETCH_LIST_SIZE2  = 1

total_size = (8+8+8*6) * MINING_PERIOD + (8 + 8 * PREFETCH_LIST_SIZE) * PREFETCH_TABLE2
number_of_block = total_size / (64*1024)
print(number_of_block)

# [CACHE_SIZE,
params['w70'] = []





'''
w70 total 6928386, unique 506904, 5800: 0.64, 30000: 0.8
cache size 52047, hit rate 0.852480, total check 3502719, prefetch 863807, hit 334150, prefetch table size 356909, ave len: 1.287743
prefetch table size 25000, low hit rate, tiny improvement

w71 total 6840777, unique 673334    cache size 57055, hit rate 0.627056, prefetch 36374, hit 4555
cache size 57055, hit rate 0.867609, total check 4554869, prefetch 2065442, hit 1799792, prefetch table size 653270, ave len: 1.109855
cache size 57055, hit rate 0.702659, total check 1544080, prefetch 540763, hit 516037, prefetch table size 90352, ave len: 1.069428

w72 total 6407186, unique 430716






w73 total 6368816, unique 601227 cache size 63687, hit rate 0.561576, prefetch 3476, hit 859
cache size 63687, hit rate 0.617140, total check 1702449, prefetch 506015, hit 402453, prefetch table size 111152, ave len: 1.129642
cache size 63687, hit rate 0.745168, total check 6649559, prefetch 2979387, hit 1816713, prefetch table size 580916, ave len: 1.501974


w74






'''