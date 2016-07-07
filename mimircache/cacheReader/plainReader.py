from mimircache.cacheReader.abstractReader import cacheReaderAbstract
import mimircache.c_cacheReader as c_cacheReader


class plainReader(cacheReaderAbstract):
    def __init__(self, file_loc, open_c_reader=True):
        super(plainReader, self).__init__(file_loc)
        self.trace_file = open(file_loc, 'r')
        if open_c_reader:
            self.cReader = c_cacheReader.setup_reader(file_loc, 'p')

    def read_one_element(self):
        super().read_one_element()
        line = self.trace_file.readline()
        if line:
            return line.strip()
        else:
            return None

    def __next__(self):  # Python 3
        super().__next__()
        element = self.trace_file.readline().strip()
        if element:
            return element
        else:
            raise StopIteration

    def __repr__(self):
        return "basic cache reader, cache trace separated by line, %s" % super().__repr__()


if __name__ == "__main__":
    reader = plainReader('../data/trace_CloudPhysics')

    # usage one: for reading all elements
    for i in reader:
        print(i)

    # usage two: best for reading one element each time
    s = reader.read_one_element()
    # s2 = next(reader)
    while s:
        print(s)
        s = reader.read_one_element()
        # s2 = next(reader)
