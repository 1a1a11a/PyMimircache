
"""
    This module provides thread function of computing heatmap dat


    Author: Juncheng Yang <peter.waynechina@gmail.com> 02/2018
"""

from PyMimircache.utils.printing import *
from PyMimircache.const import DEF_EMA_HISTORY_WEIGHT


def cal_hr_LRU(rd, last_access_dist, cache_size, start=0, end=-1, **kwargs):
    """
    calculate the hit ratio of a given range of reuse distance

    :param rd: reuse distance list
    :param last_access_dist: how far in the past is last request, absolute distance
    :param cache_size: size of cache
    :param start: the start pos of reuse distance
    :param end: the end pos of reuse distance (not included)
    :param kwargs: real_start
    :return: hit ratio
    """

    if end == -1:
        end = len(rd)

    hit_count = 0
    miss_count = 0

    # the real trace start point,
    # this differs from start only when this is used in
    # heatmap hr_st_et calculation
    real_start = kwargs.get("real_start", start)
    # print("{} {} {} {}".format(len(rd), len(last_access_dist), start, end))

    for i in range(start, end):
        if rd[i] == -1:
            # cold miss
            miss_count += 1
            continue
        if last_access_dist[i] - (i - real_start) <= 0 and rd[i] < cache_size:
            # hit
            hit_count += 1
        else:
            # miss
            miss_count += 1

    return hit_count / (end - start)


def cal_hr_list_LRU(rd, last_access_dist, cache_size, bp, bp_start_pos, **kwargs):
    """
    the computation function for heatmap hr_st_et of LRU,
    it calculates a vertical line of hit ratio

    :param rd: reuse distance list
    :param last_access_dist: how far in the past is last request, absolute distance
    :param cache_size: size of cache
    :param bp: break point list
    :param bp_start_pos: the start pos of break point
    :param kwargs: enable_ihr, ema_coef
    :return: a list of hit ratio
    """

    hr_list = []
    hit_up_to_now = 0
    req_up_to_now = 0
    enable_ihr = kwargs.get("enable_ihr", False)
    ema_coef = kwargs.get("ema_coef", DEF_EMA_HISTORY_WEIGHT)

    for i in range(bp_start_pos, len(bp)-1):
        hr = cal_hr_LRU(rd, last_access_dist, cache_size, start=bp[i], end=bp[i+1], real_start=bp[bp_start_pos])
        hit_up_to_now += int((bp[i+1] - bp[i]) * hr + 0.5)
        req_up_to_now += bp[i+1] - bp[i]
        overall_hr = hit_up_to_now / req_up_to_now
        # print("{} {} {} {}".format(bp[i], bp[i+1], hr, hit_up_to_now))

        if enable_ihr:
            hr_list.append(hr_list[-1] * ema_coef + hr * (1 - ema_coef))
        else:
            hr_list.append(overall_hr)

    return hr_list


def cal_hr_list_general(reader_class, reader_params, cache_class, cache_size, bp, bp_start_pos, **kwargs):
    """
    the computation function for heatmap hr_st_et of a general algorithm,
    it calculates a vertical line of hit ratio

    :param reader_class: the __class__ attribute of reader, this will be used to create local reader instance
    :param reader_params:   parameters for reader, used in creating local reader instance
    :param cache_class: the __class__ attr of cache, it is used for instantiate a simulated cache
    :param cache_size: size of cache
    :param bp: break point list
    :param bp_start_pos: the start pos of break point
    :param kwargs: enable_ihr, ema_coef, cache_params
    :return: a list of hit ratio
    """

    hr_list = []
    hit_up_to_now = 0
    req_up_to_now = 0
    enable_ihr = kwargs.get("enable_ihr", False)
    ema_coef = kwargs.get("ema_coef", DEF_EMA_HISTORY_WEIGHT)

    # local reader and cache
    process_reader = reader_class(**reader_params)
    cache_params = kwargs.get("cache_params", {})
    if cache_class.__name__ == "Optimal":
        cache_params["reader"] = process_reader
    process_cache = cache_class(cache_size, **cache_params)
    if cache_class.__name__ == "Optimal":
        process_cache.set_init_ts(bp[bp_start_pos])
    process_reader.skip_n_req(bp[bp_start_pos])


    for i in range(bp_start_pos, len(bp)-1):
        n_hit_interval = 0
        n_miss_interval = 0
        for j in range(bp[i], bp[i+1]):
            req = process_reader.read_one_req()
            hit = process_cache.access(req)
            if hit:
                n_hit_interval += 1
            else:
                n_miss_interval += 1
        hr = n_hit_interval / (n_hit_interval + n_miss_interval)
        hit_up_to_now += n_hit_interval
        req_up_to_now += bp[i+1] - bp[i]
        overall_hr = hit_up_to_now / req_up_to_now
        if enable_ihr:
            hr_list.append(hr_list[-1] * ema_coef + hr * (1 - ema_coef))
        else:
            hr_list.append(overall_hr)

    process_reader.close()
    return hr_list


def cal_KL(rd_list, bp, start, epsilon=0.01):
    """
    given two list of reuse distance list, calculate KL
    rd_list1 should be the True value


    :param rd_list1:
    :param rd_list2:
    :param epsilon: smoothing
    :return:
    """
    try:
        from PyMimircache.profiler.utils.KL import transform_rd_list_to_rd_cnt, cal_KL_from_rd_cnt_cdf

        KL_list = []
        rd_list1 = rd_list[bp[start] : bp[start+1]]
        rd_cnt1 = transform_rd_list_to_rd_cnt(rd_list1, percentage=True, normalize=False, cdf=True)

        for i in range(start, len(bp)-1):
            rd_list2 = rd_list[bp[i]:bp[i+1]]
            rd_cnt2 = transform_rd_list_to_rd_cnt(rd_list2, percentage=True, normalize=False, cdf=True)
            KL_list.append(cal_KL_from_rd_cnt_cdf(rd_cnt1, rd_cnt2, epsilon))

        return KL_list
    except:
        ERROR("cannot find KL module")
        return