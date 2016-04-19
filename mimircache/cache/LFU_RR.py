import random
from mimircache.cache.abstractLFU import abstractLFU


class LFU_RR(abstractLFU):
    def __init__(self, cache_size=1000):
        super(LFU_RR, self).__init__(cache_size)

    def find_evict_key(self):
        r = random.randrange(0, len(self.least_freq_elements_list))
        evict_key = self.least_freq_elements_list[r]
        self.least_freq_elements_list.remove(evict_key)
        return evict_key


if __name__ == "__main__":
    pass
