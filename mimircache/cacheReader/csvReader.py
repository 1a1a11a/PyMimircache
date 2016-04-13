import string
from mimircache.cacheReader.abstractReader import cacheReaderAbstract


class csvCacheReader(cacheReaderAbstract):
    def __init__(self, file_loc, column, header=False, delimiter=','):
        super(csvCacheReader, self).__init__(file_loc)
        self.column = column
        self.counter = 0
        self.header_bool = header
        self.delimiter = delimiter
        if header:
            self.headers = [i.strip(string.whitespace) for i in self.trace_file.readline().split(self.delimiter)]

    def read_one_element(self):
        super().read_one_element()
        line = self.trace_file.readline()
        if line:
            return line.split(self.delimiter)[self.column].strip()
        else:
            return None

    def lines(self):
        line = self.trace_file.readline()
        while line:
            line_split = line.split(self.delimiter)
            d = {}
            if self.header_bool:
                for i in range(len(self.headers)):
                    d[self.headers[i]] = line_split[i].strip(string.whitespace)
            else:
                for key, value in enumerate(line_split):
                    d[key] = value
            line = self.trace_file.readline()
            yield d

    def __next__(self):  # Python 3
        super().__next__()
        element = self.trace_file.readline()
        if element:
            return element.split(self.delimiter)[self.column].strip()
        else:
            raise StopIteration

    def __repr__(self):
        return "csv cache reader, specified column: {}, column begins from 0".format(self.column)


if __name__ == "__main__":
    reader = csvCacheReader('../data/trace_CloudPhysics_txt2', 4, header=True, delimiter=',')

    # usage one: for reading all elements
    # for i in reader:
    #     print(i)

    for i in reader.lines():
        print(i['op'])

        # usage two: best for reading one element each time
        # s = reader.read_one_element()
        # # s2 = next(reader)
        # while (s):
        #     print(s)
        #     s = reader.read_one_element()
        # s2 = next(reader)
