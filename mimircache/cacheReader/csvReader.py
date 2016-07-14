import string
import mimircache.c_cacheReader as c_cacheReader
from mimircache.cacheReader.abstractReader import cacheReaderAbstract


class csvReader(cacheReaderAbstract):
    def __init__(self, file_loc, init_params=None, open_c_reader=True):
        super(csvReader, self).__init__(file_loc)
        self.trace_file = open(file_loc, 'r')

        self.init_params = init_params
        assert "label_column" in init_params, "please provide label_column for csv reader"
        self.label_column = init_params['label_column']
        self.counter = 0

        self.header_bool = init_params.get('header')
        if 'delimiter' in init_params:
            self.delimiter = init_params.get('delimiter')
        else:
            self.delimiter = ','

        if self.header_bool:
            self.headers = [i.strip(string.whitespace) for i in self.trace_file.readline().split(self.delimiter)]
            # self.read_one_element()

        if open_c_reader:
            self.cReader = c_cacheReader.setup_reader(file_loc, 'c', init_params=init_params)

    def read_one_element(self):
        super().read_one_element()
        line = self.trace_file.readline()
        if line:
            return line.split(self.delimiter)[self.label_column].strip()
        else:
            return None

    def lines_dict(self):
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

    def lines(self):
        line = self.trace_file.readline()
        while line:
            line_split = tuple(line.split(self.delimiter))
            line = self.trace_file.readline()
            yield line_split


    def __next__(self):  # Python 3
        super().__next__()
        element = self.trace_file.readline().strip()
        if element:
            return element.split(self.delimiter)[self.label_column].strip()
        else:
            raise StopIteration

    def __repr__(self):
        return "csv cache reader {}, specified column: {}, column begins from 0".format(self.file_loc, self.label_column)




if __name__ == "__main__":
    reader = csvReader('../data/trace.csv', {"label_column":4, 'size_column':3, 'header':True, 'delimiter':','})
    # for i in range(10):
    #     print(c_cacheReader.read_one_element(reader.cReader))
    # c_cacheReader.reset_reader(reader.cReader)
    # for i in range(10):
    #     print(c_cacheReader.read_one_element(reader.cReader))

    e = c_cacheReader.read_one_element(reader.cReader)
    while e:
        print(e)
        e = c_cacheReader.read_one_element(reader.cReader)

        # usage one: for reading all elements
    # for i in reader:
    #     print(i)

    # for i in reader.lines_dict():
    #     print(i['op'])

        # usage two: best for reading one element each time
        # s = reader.read_one_element()
        # # s2 = next(reader)
        # while (s):
        #     print(s)
        #     s = reader.read_one_element()
        # s2 = next(reader)
