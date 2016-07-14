import math

from mimircache import *
from mimircache.cache.Optimal import Optimal


def calc_hit_rate_start_time_end_time_subprocess_general(order, cache, break_points_share_array, reader, q, **kwargs):
    """
    the child process for calculating hit rate for a general cache replacement algorithm,
    each child process will calculate for a column with fixed starting time
    :param cache:
    :param order: the order of column the child is working on
    :param break_points_share_array:
    :param reader:
    :param q
    :return: nothing, but add to the queue a list of result in the form of (x, y, hit_rate) with x as fixed value
    """

    # new for optimal using c_generalProfiler
    # c_generalProfiler.get_hit_rate()

    cache_size = kwargs['cache_size']
    # if cache == 'Random':
    #     c = Random(cache_size=cache_size)
    # elif cache == 'SLRU':
    #     c = SLRU(cache_size=cache_size)
    # elif cache == 'AdaptiveSLRU':
    #     c = AdaptiveSLRU(cache_size=cache_size)
    # elif cache == 'LFU_RR':
    #     c = LFU_RR(cache_size=cache_size)
    # elif cache == 'LRU':
    #     c = LRU(cache_size=cache_size)
    # elif cache == "optimal":
    #     c = optimal(cache_size, reader)
    # elif cache == 'FIFO':
    #     c = FIFO(cache_size=cache_size)
    # else:
    if 'cache_params' in kwargs and kwargs['cache_params']:
        c = cache(cache_size=cache_size, cache_params=kwargs['cache_params'])
    else:
        c = cache(cache_size=cache_size)


    result_list = []
    total_hc = 0  # total hit count
    total_mc = 0  # total miss count
    pos_in_break_points = order + 1
    line_num = 0
    # if type(reader) != plainCacheReader:
    #     reader_new = plainCacheReader('temp.dat')
    # else:
    #     reader_new = type(reader)(reader.file_loc)

    if isinstance(reader, csvReader):
        reader_new = type(reader)(reader.file_loc, init_params=reader.init_params, open_c_reader=False)
    else:
        reader_new = type(reader)(reader.file_loc, open_c_reader=False)

    # TODO: figure out line size here and add seek method in reader base class
    # TODO: use mmap here to improve performance

    for line in reader_new:
        if line_num < break_points_share_array[order]:
            line_num += 1
            continue
        # fix this hack
        if cache == Optimal:
            c.ts = line_num

        line_num += 1
        if c.addElement(line):
            total_hc += 1
        else:
            total_mc += 1

        if pos_in_break_points < len(break_points_share_array) and \
                        line_num == break_points_share_array[pos_in_break_points]:
            hr = total_hc / (break_points_share_array[pos_in_break_points] - break_points_share_array[order])
            result_list.append((order, pos_in_break_points - 1, hr))
            pos_in_break_points += 1
            # print("{}: {}".format(total_hc, total_mc))
    q.put(result_list)
    reader_new.close()


# LRU

def calc_hit_rate_start_time_cache_size_subprocess(order, break_points_share_array, reuse_dist_share_array, q,
                                                   **kwargs):
    """
    the child process for calculating hit rate of different cache size with different starting time, but fixed end time,
    and each child process will calculate for a column with a given starting time
    :param q:
    :param reuse_dist_share_array:
    :param break_points_share_array:
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, hit_rate) with x as fixed value(starting time), y as cache size
    """

    max_rd = kwargs['max_rd']
    bin_size = kwargs['bin_size']

    result_list = []

    rd_distribution = [0] * int(max_rd // bin_size + 1)

    for i in range(break_points_share_array[order], break_points_share_array[-1]):
        rd_distribution[reuse_dist_share_array[i] // bin_size] += 1

    num_of_total_request = break_points_share_array[-1] - break_points_share_array[order]

    accum = 0
    for i in range(len(rd_distribution)):
        accum += rd_distribution[i]
        result_list.append((order, i, accum / num_of_total_request))

    q.put(result_list)


def _hit_rate_start_time_end_time_calc_hit_count(reuse_dist_array, last_access_array, cache_size, begin_pos, end_pos,
                                                 real_start, **kwargs):
    """
    called by hit_rate_start_time_end_time to calculate hit count
    :rtype: count of hit
    :param reuse_dist_array:
    :param cache_size:
    :param begin_pos:
    :param end_pos: end pos of trace in current partition (not included)
    :param real_start: the real start position of cache trace
    :return:
    """

    hit_count = 0
    miss_count = 0
    try:
        for i in range(begin_pos, end_pos):
            if reuse_dist_array[i] == -1:
                # never appear
                miss_count += 1
                continue
            if last_access_array[i] - (i - real_start) <= 0 and reuse_dist_array[i] < cache_size:
                # hit
                hit_count += 1
            else:
                # miss
                miss_count += 1
    except:
        print("error: begin: {}, end: {}, i: {}".format(begin_pos, end_pos, i))
    return hit_count


def calc_hit_rate_start_time_end_time_subprocess(order, break_points_share_array, reuse_dist_share_array, q,
                                                 **kwargs):
    """
    the child process for calculating hit rate, each child process will calculate for
    a column with fixed starting time
    :param q:
    :param reuse_dist_share_array:
    :param break_points_share_array:
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, hit_rate) with x as fixed value
    """

    cache_size = kwargs['cache_size']
    last_access_array = kwargs['last_access_array']

    result_list = []
    total_hc = 0
    for i in range(order + 1, len(break_points_share_array)):
        hc = _hit_rate_start_time_end_time_calc_hit_count(reuse_dist_share_array, last_access_array, cache_size,
                                                          break_points_share_array[i - 1],
                                                          break_points_share_array[i],
                                                          break_points_share_array[order])
        total_hc += hc
        hr = total_hc / (break_points_share_array[i] - break_points_share_array[order])
        result_list.append((order, i - 1, hr))
    q.put(result_list)


def calc_avg_rd_start_time_end_time_subprocess(order, break_points_share_array, reuse_dist_share_array, q, **kwargs):
    """
    the child process for calculating average reuse distance in each block, each child process will calculate for
    a column with fixed starting time
    :param q:
    :param reuse_dist_share_array:
    :param break_points_share_array:
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, hit_rate) with x as fixed value
    """

    result_list = []
    rd = 0
    never_see = 0
    count = 0
    for i in range(order + 1, len(break_points_share_array)):
        for j in range(break_points_share_array[i - 1], break_points_share_array[i]):
            if reuse_dist_share_array[j] != -1:
                rd += reuse_dist_share_array[j]
                count += 1
            else:
                never_see += 1
        if break_points_share_array[i] - break_points_share_array[order] - never_see != 0:
            result_list.append(
                (order, i - 1, rd / (break_points_share_array[i] - break_points_share_array[order] - never_see)))
    q.put(result_list)


def calc_cold_miss_count_start_time_end_time_subprocess(order, break_points_share_array, reuse_dist_share_array, q,
                                                        **kwargs):
    """
    the child process for calculating cold miss count in each block, each child process will calculate for
    a column with fixed starting time
    :param q:
    :param reuse_dist_share_array:
    :param break_points_share_array:
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, miss_count) with x as fixed value
    """

    result_list = []
    rd = 0
    never_see = 0
    count = 0
    for i in range(order + 1, len(break_points_share_array)):
        for j in range(break_points_share_array[i - 1], break_points_share_array[i]):
            if reuse_dist_share_array[j] != -1:
                rd += reuse_dist_share_array[j]
                count += 1
            else:
                never_see += 1

        result_list.append((order, i - 1, never_see))

    q.put(result_list)


def calc_rd_distribution_subprocess_deprecated(order, break_points_share_array, reuse_dist_share_array, q, **kwargs):
    """
    deprecated
    For LRU
    the child process for calculating average reuse distance in each block, each child process will calculate for
    a column with fixed starting time
    :param q:
    :param reuse_dist_share_array:
    :param break_points_share_array:
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, hit_rate) with x as fixed value
    """

    LOG_NUM = kwargs['log_num']
    max_rd = kwargs['max_rd']

    result_list = []
    rd_bucket = [0] * int(math.log(max_rd, LOG_NUM) + 1)

    for j in range(break_points_share_array[order], break_points_share_array[order + 1]):
        if reuse_dist_share_array[j] != -1:
            if reuse_dist_share_array[j] == 0:
                rd_bucket[0] += 1
            else:
                rd_bucket[int(math.log(reuse_dist_share_array[j], LOG_NUM))] += 1

    for i in range(len(rd_bucket)):
        result_list.append((order, i - 1, rd_bucket[i]))

    q.put(result_list)
