from mimircache.cacheReader.abstractReader import cacheReaderAbstract


class plainCacheReader(cacheReaderAbstract):
    def __init__(self, file_loc):
        super(plainCacheReader, self).__init__(file_loc)

    def read_one_element(self):
        super().read_one_element()
        line = self.trace_file.readline()
        if line:
            return line.strip()
        else:
            return None

    def __next__(self):  # Python 3
        super().__next__()
        element = self.trace_file.readline()
        if element:
            return element.strip()
        else:
            raise StopIteration

    def __repr__(self):
        return "basic cache reader, cache trace separated by line, %s" % super().__repr__()


if __name__ == "__main__":
    reader = plainCacheReader('../data/trace_CloudPhysics')

    # usage one: for reading all elements
    for i in reader:
        print(i)

    # usage two: best for reading one element each time
    s = reader.read_one_element()
    # s2 = next(reader)
    while (s):
        print(s)
        s = reader.read_one_element()
        # s2 = next(reader)
