def calc_cold_miss_count(break_points, reader, **kwargs):
    """
    the function for calculating cold miss count in each block
    :param reader:
    :param break_points:
    :return:
    """

    result_list = [0] * (len(break_points) - 1)
    seen_set = set()
    for i in range(len(break_points) - 1):
        never_see = 0
        for j in range(break_points[i + 1] - break_points[i]):
            r = next(reader)
            if r not in seen_set:
                seen_set.add(r)
                never_see += 1
        result_list[i] = never_see

    return result_list
