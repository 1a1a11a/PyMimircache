# coding=utf-8


# from mimircache import
from heapdict import heapdict
import heapq


class multiReader:
    def __init__(self, readers, reading_type="real_time"):
        self.readers = readers
        self.reading_type = reading_type
        assert self.reading_type in ["real_time", "virtual_time"], "allowed reading_type: real_time, virtual_time"
        if len(self.readers) == 0:
            raise RuntimeError("reader list is empty")

        self.num_of_read = 0
        # self.reader_buffer = heapdict()
        self.reader_buffer = []

        if self.reading_type == "real_time":
            for i in range(len(self.readers)):
                t, req = self.readers[i].read_time_request()
                self.reader_buffer.append((t, i, req))
                self.num_of_read += 1

        elif self.reading_type == "virtual_time":
            for i in range(len(self.readers)):
                req = self.readers[i].read_one_element()
                # self.reader_buffer[(i, req)] = self.num_of_read
                self.reader_buffer.append((self.num_of_read, i, req))
                self.num_of_read += 1

        heapq.heapify(self.reader_buffer)


    def read_one_element(self):
        element = self.read_with_readerID()
        if element is not None:
            return element[1]
        else:
            return None


    def read_with_readerID(self):
        # item, pri = self.reader_buffer.popitem()
        # pos, req = item

        item = heapq.heappop(self.reader_buffer)
        t_old, pos, req_old = item

        reader = self.readers[pos]

        if self.reading_type == "real_time":
            item = reader.read_time_request()
            if item is not None:
                t, req = item
                # self.reader_buffer[(pos, req)] = t
                heapq.heappush(self.reader_buffer, (t, pos, req))
            else:
                return None

        elif self.reading_type == "virtual_time":
            req = reader.read_one_element()
            if req is not None:
                # self.reader_buffer[(pos, req)] = self.num_of_read
                heapq.heappush(self.reader_buffer, (self.num_of_read, pos, req))
            else:
                return None

        self.num_of_read += 1
        return pos, req_old



    def reset(self):
        for reader in self.readers:
            reader.reset()
        self.num_of_read = 0

    def close_all_readers(self):
        for reader in self.readers:
            reader.close()


    def __iter__(self):
        return self


    def __next__(self):
        v = self.read_one_element()
        if v is not None:
            return v
        else:
            raise StopIteration


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return "a multi reader using real_time as reading order"



if __name__ == "__main__":
    from mimircache import *
    from mimircache.bin.conf import *

    reader_list = []
    for i in range(1, 10):
        reader_list.append(csvReader("/home/jason/ALL_DATA/Akamai/day/2016100{}.sort".format(i),
                                     init_params=AKAMAI_CSV))
    mReader = multiReader(reader_list) #, reading_type="virtual_time")

    for n, i in enumerate(mReader):
        print(i)
        if n>10000:
            break