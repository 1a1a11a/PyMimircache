# coding=utf-8
import string
from mimircache.const import CExtensionMode
if CExtensionMode:
    import mimircache.c_cacheReader as c_cacheReader
from mimircache.cacheReader.abstractReader import cacheReaderAbstract


class csvReader(cacheReaderAbstract):
    def __init__(self, file_loc, data_type='c', init_params=None, open_c_reader=True):
        super(csvReader, self).__init__(file_loc, data_type)
        assert init_params is not None, "please provide init_param for csvReader"
        assert "label_column" in init_params, "please provide label_column for csv reader"

        self.trace_file = open(file_loc, 'r', encoding='utf-8', errors='ignore')
        self.init_params = init_params
        self.label_column = init_params['label_column']
        self.time_column = init_params.get("real_time_column", -1)

        self.header_bool = init_params.get('header', False)
        self.delimiter = init_params.get('delimiter', ',')

        if self.header_bool:
            self.headers = [i.strip(string.whitespace) for i in self.trace_file.readline().split(self.delimiter)]
            self.read_one_element()

        if open_c_reader:
            self.cReader = c_cacheReader.setup_reader(file_loc, 'c', data_type=data_type, init_params=init_params)


    def read_one_element(self):
        super().read_one_element()
        line = self.trace_file.readline()
        while line and len(line.strip())==0:
            line = self.trace_file.readline()

        if line:
            return line.split(self.delimiter)[self.label_column -1].strip()
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

    def read_time_request(self):
        """
        return real_time information for the request in the form of (time, request)
        :return:
        """
        super().read_one_element()
        line = self.trace_file.readline()
        if line:
            line = line.split(self.delimiter)
            try:
                return float(line[self.time_column - 1].strip()), line[self.label_column - 1].strip()
            except Exception as e:
                print("ERROR reading data: {}, current line: {}".format(e, line))

        else:
            return None


    def __next__(self):  # Python 3
        super().__next__()
        element = self.read_one_element()
        if element is not None:
            return element
        else:
            raise StopIteration

    def __repr__(self):
        return "csv cache reader {}, key column: {}, column begins from 1".format(self.file_loc, self.label_column)

