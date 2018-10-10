# coding=utf-8
"""
    a csv trace reader

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/06

"""
import string
from PyMimircache.const import ALLOW_C_MIMIRCACHE, INSTALL_PHASE
from PyMimircache.utils.printing import *

if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.CacheReader as c_cacheReader
from PyMimircache.cacheReader.abstractReader import AbstractReader


class CsvReader(AbstractReader):
    """
    CsvReader class
    """
    all = ["read_one_req", "read_complete_req", "lines_dict",
           "lines", "read_time_req", "reset", "copy", "get_params"]

    def __init__(self, file_loc,
                 data_type='c',
                 init_params=None,
                 block_unit_size=0,
                 disk_sector_size=0,
                 open_c_reader=True,
                 **kwargs):
        """
        :param file_loc:            location of the file
        :param data_type:           type of data, can be "l" for int/long, "c" for string
        :param init_params:         the init_params for opening csv
        :param block_unit_size:     block size for storage system, 0 when disabled
        :param disk_sector_size:    size of disk sector
        :param open_c_reader:       bool for whether open reader in C backend
        :param kwargs:              not used now
        """

        super(CsvReader, self).__init__(file_loc, data_type, block_unit_size, disk_sector_size,
                                        open_c_reader, kwargs.get("lock", None))
        assert init_params is not None, "please provide init_param for csvReader"
        assert "label" in init_params, "please provide label for csv reader"

        self.trace_file = open(file_loc, 'rb')
        # self.trace_file = open(file_loc, 'r', encoding='utf-8', errors='ignore')
        self.init_params = init_params
        self.label_column = init_params['label']
        self.time_column = init_params.get("real_time", )
        self.size_column = init_params.get("size", )

        if self.time_column != -1:
            self.support_real_time = True

        if self.size_column != -1:
            self.support_size = True

        if block_unit_size != 0:
            assert "size" in init_params, "please provide size_column option to consider request size"

        self.header_bool = init_params.get('header', )
        self.delimiter = init_params.get('delimiter', ",")
        if "delimiter" not in init_params:
            INFO("open {} using default delimiter \",\" for CsvReader".format(file_loc))


        if self.header_bool:
            self.headers = [i.strip(string.whitespace) for i in
                            self.trace_file.readline().decode().split(self.delimiter)]
            # self.trace_file.readline()

        if ALLOW_C_MIMIRCACHE and open_c_reader:
            self.c_reader = c_cacheReader.setup_reader(file_loc, 'c', data_type=data_type,
                                                                 block_unit_size=block_unit_size,
                                                                 disk_sector_size=disk_sector_size,
                                                                 init_params=init_params)

    def read_one_req(self):
        """
        read one request, return the lbn/objID 
        :return: 
        """
        super().read_one_req()

        line = self.trace_file.readline().decode('utf-8', 'ignore')
        while line and len(line.strip()) == 0:
            line = self.trace_file.readline().decode()
        if line:
            ret = line.split(self.delimiter)[self.label_column - 1].strip()
            if self.data_type == 'l':
                ret = int(ret)
                if self.block_unit_size != 0 and self.disk_sector_size != 0:
                    ret = ret * self.disk_sector_size // self.block_unit_size
            return ret
        else:
            return None

    def read_complete_req(self):
        """
        read the complete line, including request and its all related info
        :return: a list of all info of the request
        """

        super().read_one_req()

        line = self.trace_file.readline().decode()
        while line and len(line.strip()) == 0:
            line = self.trace_file.readline().decode()
        if line:
            line_split = line.strip().split(self.delimiter)
            if self.block_unit_size != 0 and self.disk_sector_size != 0:
                line_split[self.label_column - 1] = line_split[self.label_column - 1] * \
                                                    self.disk_sector_size // self.block_unit_size
            return line_split
        else:
            return None

    def lines_dict(self):
        """
        return a dict with column header->data 
        note this function does not convert lbn even if disk_sector_size and block_unit_size are set 
        :return: 
        """
        line = self.trace_file.readline().decode()
        while line and len(line.strip()) == 0:
            line = self.trace_file.readline().decode()

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
            # raise StopIteration

    def lines(self):
        """
        a generator for reading all the information of current request/line
        :return: a tuple of current request
        """
        line = self.trace_file.readline().decode()
        while line and len(line.strip()) == 0:
            line = self.trace_file.readline().decode()

        while line:
            line_split = tuple(line.split(self.delimiter))
            line = self.trace_file.readline()
            yield line_split
            # raise StopIteration

    def read_time_req(self):
        """
        return real_time information for the request in the form of (time, request)
        :return:
        """
        super().read_one_req()
        line = self.trace_file.readline().strip().decode()
        while line and len(line.strip()) == 0:
            line = self.trace_file.readline().decode()

        if line:
            line = line.split(self.delimiter)
            try:
                time = float(line[self.time_column - 1].strip())
                lbn = line[self.label_column - 1].strip()
                if self.data_type == 'l':
                    lbn = int(lbn)
                    if self.block_unit_size != 0 and self.disk_sector_size != 0:
                        lbn = lbn * self.disk_sector_size // self.block_unit_size

                return time, lbn
            except Exception as e:
                print("ERROR csvReader reading data: {}, current line: {}".format(e, line))

        else:
            return None

    def skip_n_req(self, n):
        """
        skip N requests from current position

        :param n: the number of requests to skip
        """

        for i in range(n):
            self.read_one_req()

    def reset(self):
        """
        reset reader to initial state
        :return:
        """
        super().reset()
        if self.header_bool:
            self.trace_file.readline()

    def copy(self, open_c_reader=False):
        """
        reader a deep copy of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_c_reader: whether open_c_reader_or_not, default not open
        :return: a copied reader
        """

        return CsvReader(self.file_loc, self.data_type, self.init_params,
                         self.block_unit_size, self.disk_sector_size, open_c_reader, lock=self.lock)


    def get_params(self):
        """
        return all the parameters for this reader instance in a dictionary
        :return: a dictionary containing all parameters
        """

        return {
            "file_loc": self.file_loc,
            "init_params": self.init_params,
            "data_type": self.data_type,
            "block_unit_size": self.block_unit_size,
            "disk_sector_size": self.disk_sector_size,
            "open_c_reader": self.open_c_reader,
            "lock": self.lock
        }

    def __next__(self):  # Python 3
        super().__next__()
        element = self.read_one_req()
        if element is not None:
            return element
        else:
            raise StopIteration

    def __repr__(self):
        return "csvReader for trace {}".format(self.file_loc)
