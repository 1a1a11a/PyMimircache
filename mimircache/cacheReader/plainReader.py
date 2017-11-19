# coding=utf-8
from mimircache.cacheReader.abstractReader import cacheReaderAbstract
from mimircache.const import CExtensionMode
if CExtensionMode:
    import mimircache.c_cacheReader


class plainReader(cacheReaderAbstract):
    def __init__(self, file_loc, data_type='c', open_c_reader=True):
        """
        :param file_loc:            location of the file
        :param data_type:           type of data, can be "l" for int/long, "c" for string
        :param open_c_reader:       bool for whether open reader in C backend
        """
        super(plainReader, self).__init__(file_loc, data_type, 0, 0)
        self.trace_file = open(file_loc, 'r')
        if open_c_reader:
            self.cReader = mimircache.c_cacheReader.setup_reader(file_loc, 'p', data_type=data_type, block_unit_size=0)

    def read_one_element(self):
        """
        read one request
        :return:
        """
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
        return "plainTextReader on file {}".format(self.file_loc)
