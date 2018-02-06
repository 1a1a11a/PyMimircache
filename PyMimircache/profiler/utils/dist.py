

from collections import deque



def get_last_access_dist(reader):
    """
    calculate the distance from last access, absolute distance, -1 if there is no last access

    :param reader: reader for data input
    :return: a list of dist since last access
    """

    reader.reset()
    last_access_dist = []
    last_access_time = {}
    for n, r in enumerate(reader):
        if r in last_access_time:
            last_access_dist.append(n - last_access_time[r])
        else:
            last_access_dist.append(-1)
        last_access_time[r] = n
    reader.reset()
    return last_access_dist




def get_next_access_dist(reader):
    """
    calculate the distance to next access, absolute distance, -1 if there is no next access

    :param reader: reader for data input
    :return: a list of dist to next access
    """

    reader.reset()
    reversed_dat = deque()
    next_access_dist = []
    next_access_time = {}

    # FIXME: try to do this in a more memory-efficient way

    for req in reader:
        reversed_dat.appendleft(req)

    for n, r in enumerate(reversed_dat):
        if r in next_access_time:
            next_access_dist.append(n - next_access_time[r])
        else:
            next_access_dist.append(-1)
        next_access_time[r] = n
    next_access_dist.reverse()
    reader.reset()
    return next_access_dist