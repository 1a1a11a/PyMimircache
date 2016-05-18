def _calc_hit_count_general(reuse_dist_array, cache_size, begin_pos, end_pos, real_start):
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


def _calc_hit_rate_subprocess_general(order, cache_size, break_points_share_array, reader, q):
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

    result_list = []
    total_hc = 0
    pos_in_break_points = 0

    for i in range(break_points_share_array[order], len(break_points_share_array)):

        for j in range()
            hc = _calc_hit_count_general(reuse_dist_share_array, cache_size, break_points_share_array[i - 1],
                                         break_points_share_array[i], break_points_share_array[order])
        total_hc += hc
        hr = total_hc / (break_points_share_array[i] - break_points_share_array[order])
        result_list.append((order, i, hr))
    q.put(result_list)
    # return result_list
