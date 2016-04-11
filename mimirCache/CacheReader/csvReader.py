from mimirCache.CacheReader.readerAbstract import cacheReaderAbstract


class csvCacheReader(cacheReaderAbstract):
    def __init__(self, file_loc, column):
        super(csvCacheReader, self).__init__(file_loc)
        self.column = column
        self.counter = 0

    def read_one_element(self):
        super().read_one_element()
        line = self.trace_file.readline()
        if line:
            return line.strip().split()[self.column]
        else:
            return None

    def __next__(self):  # Python 3
        super().__next__()
        element = self.trace_file.readline()
        if element:
            return element.split()[self.column]
        else:
            raise StopIteration


if __name__ == "__main__":
    reader = csvCacheReader('../Data/trace_CloudPhysics')

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
