from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cache.FIFO import FIFO
from mimircache.cache.RR import RR
import math


def _hit_rate_start_time_end_time_calc_hit_count_general(reuse_dist_array, cache_size, begin_pos, end_pos, real_start,
                                                         **kargs):
    """

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
    for i in range(begin_pos, end_pos):
        if reuse_dist_array[i] == -1:
            # never appear
            miss_count += 1
            continue
        if reuse_dist_array[i] - (i - real_start) < 0 and reuse_dist_array[i] < cache_size:
            # hit
            hit_count += 1
        else:
            # miss
            miss_count += 1
    return hit_count


def calc_hit_rate_start_time_end_time_subprocess_general(order, break_points_share_array, reader, q, **kargs):
    """
    the child process for calculating hit rate for a general cache replacement algorithm,
    each child process will calculate for a column with fixed starting time
    :param order: the order of column the child is working on
    :param cache_size:
    :param break_points_share_array:
    :param reader:
    :param q
    :return: nothing, but add to the queue a list of result in the form of (x, y, hit_rate) with x as fixed value
    """

    cache_size = kargs['cache_size']

    result_list = []
    total_hc = 0  # total hit count
    total_mc = 0  # total miss count
    pos_in_break_points = order + 1
    line_num = 0
    if type(reader) != plainCacheReader:
        reader_new = plainCacheReader('temp.dat')
    else:
        reader_new = type(reader)(reader.file_loc)
    c = RR(cache_size=cache_size)
    # for i in range(break_points_share_array[order], ):
    # TODO: figure out line size here and add seek method in reader base class
    # TODO: use mmap here to improve performance
    for line in reader_new:
        if line_num < break_points_share_array[order]:
            line_num += 1
            continue
        line_num += 1
        if c.addElement(line):
            total_hc += 1
        else:
            total_mc += 1

        if pos_in_break_points < len(break_points_share_array) and \
                        line_num == break_points_share_array[pos_in_break_points]:
            hr = total_hc / (break_points_share_array[pos_in_break_points] - break_points_share_array[order])
            result_list.append((order, pos_in_break_points, hr))
            pos_in_break_points += 1
    q.put(result_list)
    # return result_list


def _hit_rate_start_time_end_time_calc_hit_count(reuse_dist_array, begin_pos, end_pos, real_start, **kargs):
    """

    :rtype: count of hit
    :param reuse_dist_array:
    :param cache_size:
    :param begin_pos:
    :param end_pos: end pos of trace in current partition (not included)
    :param real_start: the real start position of cache trace
    :return:
    """

    cache_size = kargs['cache_size']

    hit_count = 0
    miss_count = 0
    for i in range(begin_pos, end_pos):
        if reuse_dist_array[i] == -1:
            # never appear
            miss_count += 1
            continue
        if reuse_dist_array[i] - (i - real_start) < 0 and reuse_dist_array[i] < cache_size:
            # hit
            hit_count += 1
        else:
            # miss
            miss_count += 1
    return hit_count


def calc_hit_rate_start_time_cache_size_subprocess(order, break_points_share_array, reuse_dist_share_array, q, **kargs):
    """
    the child process for calculating hit rate of different cache size with different starting time, but fixed end time,
    and each child process will calculate for a column with a given starting time
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, hit_rate) with x as fixed value(starting time), y as cache size
    """

    max_rd = kargs['max_rd']

    result_list = []
    total_hc = 0

    rd_distribution = [0] * (max_rd + 1)

    for i in range(break_points_share_array[order], break_points_share_array[-1]):
        rd_distribution[reuse_dist_share_array[i]] += 1

    num_of_total_request = break_points_share_array[-1] - break_points_share_array[order]

    accum = 0
    for i in range(len(rd_distribution)):
        accum += rd_distribution[i]
        result_list.append((order, i, accum / num_of_total_request))

    q.put(result_list)


def calc_hit_rate_start_time_end_time_subprocess(order, cache_size, break_points_share_array, reuse_dist_share_array, q,
                                                 **kargs):
    """
    the child process for calculating hit rate, each child process will calculate for
    a column with fixed starting time
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, hit_rate) with x as fixed value
    """

    cache_size = kargs['cache_size']

    result_list = []
    total_hc = 0
    for i in range(order + 1, len(break_points_share_array)):
        hc = _hit_rate_start_time_end_time_calc_hit_count(reuse_dist_share_array, cache_size,
                                                          break_points_share_array[i - 1],
                                                          break_points_share_array[i], break_points_share_array[order])
        total_hc += hc
        hr = total_hc / (break_points_share_array[i] - break_points_share_array[order])
        result_list.append((order, i, hr))
    q.put(result_list)
    # return result_list


def calc_avg_rd_start_time_end_time_subprocess(order, break_points_share_array, reuse_dist_share_array, q, **kargs):
    """
    the child process for calculating average reuse distance in each block, each child process will calculate for
    a column with fixed starting time
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
        # result_list.append((order, i, rd / (break_points_share_array[i] - break_points_share_array[order])))
        if break_points_share_array[i] - break_points_share_array[order] - never_see != 0:
            result_list.append(
                (order, i, rd / (break_points_share_array[i] - break_points_share_array[order] - never_see)))
    q.put(result_list)


def calc_cold_miss_count_start_time_end_time_subprocess(order, break_points_share_array, reuse_dist_share_array, q,
                                                        **kargs):
    """
    the child process for calculating cold miss count in each block, each child process will calculate for
    a column with fixed starting time
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

        result_list.append((order, i, never_see))

    q.put(result_list)


def calc_rd_distribution_subprocess(order, break_points_share_array, reuse_dist_share_array, q, **kargs):
    """
    For LRU
    the child process for calculating average reuse distance in each block, each child process will calculate for
    a column with fixed starting time
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, hit_rate) with x as fixed value
    """

    LOG_NUM = kargs['log_num']
    max_rd = kargs['max_rd']

    result_list = []
    # rd_bucket = [0]* HEATMAP_GRADIENT
    rd_bucket = [0] * int(math.log(max_rd, LOG_NUM) + 1)
    # never_see = 0
    # count = 0
    # gap = max_rd//HEATMAP_GRADIENT+1


    for j in range(break_points_share_array[order], break_points_share_array[order + 1]):
        if reuse_dist_share_array[j] != -1:
            if reuse_dist_share_array[j] == 0:
                rd_bucket[0] += 1
            else:
                rd_bucket[int(math.log(reuse_dist_share_array[j], LOG_NUM))] += 1

                # if reuse_dist_share_array[j] // gap < len(rd_bucket):
                #     rd_bucket[reuse_dist_share_array[j] // gap] += 1
                # else:
                #     rd_bucket[-1] += 1
    for i in range(len(rd_bucket)):
        result_list.append((order, i, rd_bucket[i]))

        # if rd_bucket[i]:
        #     result_list.append((order, i, math.log2(rd_bucket[i])))
        # else:
        #     result_list.append((order, i, 0))
    q.put(result_list)
