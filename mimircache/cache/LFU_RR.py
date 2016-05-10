'''

'''

# still CPU intensive, needs to find out why and optimize


import random
from mimircache.cache.abstractLFU import abstractLFU


class LFU_RR(abstractLFU):
    def __init__(self, cache_size=1000):
        super(LFU_RR, self).__init__(cache_size)

    def find_evict_key(self):
        r = random.randrange(0, len(self.least_freq_elements_list))
        evict_key = self.least_freq_elements_list[r]
        count = 0
        while not evict_key:
            r = random.randrange(0, len(self.least_freq_elements_list))
            evict_key = self.least_freq_elements_list[r]
            count += 1
        self.least_freq_elements_list[r] = None

        if count > 10:
            new_list = [e for e in self.least_freq_elements_list if e]
            del self.least_freq_elements_list
            self.least_freq_elements_list = new_list

        return evict_key


if __name__ == "__main__":
    pass
