# coding=utf-8

"""
    binaryReader for reading binary trace

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/06

"""

import io
import os
import struct

from PyMimircache.const import ALLOW_C_MIMIRCACHE, INSTALL_PHASE

if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.CacheReader as c_cacheReader
from PyMimircache.cacheReader.abstractReader import AbstractReader


class BinaryReader(AbstractReader):
    """
    BinaryReader class for reading binary trace
    """
    all = ["read_one_req", "read_complete_req", "get_num_of_req", "skip_n_req",
           "lines", "read_time_req", "reset", "copy", "get_params"]

    def __init__(self, file_loc, init_params, data_type='c',
                 block_unit_size=0, disk_sector_size=0, open_c_reader=True, **kwargs):
        """
        initialization function for binaryReader

        the init_params specify the parameters for opening the trace, it is a dictionary of following key-value pairs

        +------------------+--------------+---------------------+---------------------------------------------------+
        | Keyword Argument | Value Type   | Default Value       | Description                                       |
        +==================+==============+=====================+===================================================+
        | label            | int          | this is required    | the column of the label of the request            |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | fmt              | string       | this is required    | fmt string of binary data, same as python struct  |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | real_time        | int          |        NA           | the column of real time                           |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | op               | int          |        NA           | the column of operation (read/write)              |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | size             | int          |        NA           | the column of block/request size                  |
        +------------------+--------------+---------------------+---------------------------------------------------+


        :param file_loc:            location of the file
        :param init_params:         init_params for binaryReader, see above
        :param data_type:           type of data(label), can be "l" for int/long, "c" for string
        :param block_unit_size:     block size for storage system, 0 when disabled
        :param disk_sector_size:    size of disk sector
        :param open_c_reader:       whether open c reader
        :param kwargs:              not used now
        """

        super(BinaryReader, self).__init__(file_loc, data_type, block_unit_size, disk_sector_size,
                                           open_c_reader, kwargs.get("lock", None))
        assert 'fmt' in init_params, "please provide format string(fmt) in init_params"
        assert "label" in init_params, "please specify the order of label, beginning from 1"
        if block_unit_size != 0:
            assert "size" in init_params, "please provide size option to consider request size"

        self.init_params = init_params
        self.fmt = init_params['fmt']
        # this number begins from 1, so need to reduce by one before use
        self.label_column = init_params['label']
        self.time_column = init_params.get("real_time", )
        self.size_column = init_params.get("size", )

        self.trace_file = open(file_loc, 'rb')
        self.struct_instance = struct.Struct(self.fmt)
        self.record_size = struct.calcsize(self.fmt)
        self.trace_file_size = os.path.getsize(self.file_loc)
        assert self.trace_file_size % self.record_size == 0, \
            "file size ({}) is not multiple of record size ({})".format(self.trace_file_size, self.record_size)

        if self.time_column != -1:
            self.support_real_time = True
        if self.size_column != -1:
            self.support_size = True

        if ALLOW_C_MIMIRCACHE and open_c_reader:
            # the data type here is not real data type, it will auto correct in C
            self.c_reader = c_cacheReader.setup_reader(file_loc, 'b', data_type=self.data_type,
                                                                 block_unit_size=block_unit_size,
                                                                 disk_sector_size=disk_sector_size,
                                                                 init_params=init_params)
        self.get_num_of_req()

        # this might cause problem on Windows and it does give any performance gain
        # self.mm = mmap.mmap(self.trace_file.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
        # self.trace_file_original = self.trace_file
        # self.trace_file = self.mm

    def get_num_of_req(self):
        """
        count the number of requests in the trace, fast for binary type trace,
        for plain/csv type trace, this is slow
        :return: the number of requests in the trace
        """

        if self.num_of_req > 0:
            return self.num_of_req

        self.num_of_req = self.trace_file_size // self.record_size
        return self.num_of_req

    def read_one_req(self):
        """
        read one request, only return the label of the request
        :return: the label of request
        """

        super().read_one_req()

        b = self.trace_file.read(self.record_size)
        if b and len(b):
            ret = self.struct_instance.unpack(b)[self.label_column - 1]
            if self.data_type == 'l':
                if ret and self.block_unit_size != 0 and self.disk_sector_size != 0:
                    ret = int(ret) * self.disk_sector_size // self.block_unit_size
                return ret
            else:
                return ret
        else:
            return None

    def read_complete_req(self):
        """
        read the complete line, including request and its all related info
        :return: a list of all info in the request
        """

        super().read_one_req()

        b = self.trace_file.read(self.record_size)
        if len(b):
            ret = list(self.struct_instance.unpack(b))
            if self.block_unit_size != 0 and self.disk_sector_size != 0:
                ret[self.label_column - 1] = ret[self.label_column - 1] * self.disk_sector_size // self.block_unit_size
            return ret
        else:
            return None


    def read_last_req(self, label_only=False, max_req_size=102400):
        """
        read the complete line, including request and its all related info
        :return: a list of all info in the request
        """

        return super().read_last_req(label_only, max_req_size=self.record_size)


    def lines(self):
        """
        a generator for reading the complete request (including label and other information)
        similar to read_complete_req, but this functions is generator
        :return: a list of information of current request
        """

        b = self.trace_file.read(self.record_size)
        while len(b):
            ret = list(self.struct_instance.unpack(b))
            if self.block_unit_size != 0 and self.disk_sector_size != 0:
                ret[self.label_column - 1] = ret[self.label_column - 1] * self.disk_sector_size // self.block_unit_size
            b = self.trace_file.read(self.record_size)
            yield ret

    def read_time_req(self):
        """
        return real_time information for the request in the form of (time, request)
        :return: a tuple of (time, request label)
        """

        assert self.time_column != -1, "you need to provide time in order to use this function"
        super().read_one_req()

        b = self.trace_file.read(self.record_size)
        if len(b):
            ret = self.struct_instance.unpack(b)
            try:
                time = float(ret[self.time_column - 1])
                obj = ret[self.label_column - 1]
                if self.data_type == 'l':
                    if self.block_unit_size != 0 and self.disk_sector_size != 0:
                        obj = int(obj) * self.disk_sector_size // self.block_unit_size
                    return time, obj
                else:
                    return time, obj
            except Exception as e:
                print("ERROR binaryReader reading data: {}, current line: {}".format(e, ret))

        else:
            return None

    def skip_n_req(self, n):
        """
        skip N requests from current position

        :param n: the number of requests to skip
        """

        self.trace_file.seek(struct.calcsize(self.fmt) * n, io.SEEK_CUR)

    def copy(self, open_c_reader=False):
        """
        reader a deep copy of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_c_reader: whether open_c_reader_or_not, default not open
        :return: a copied reader
        """

        return BinaryReader(self.file_loc, self.init_params, data_type=self.data_type,
                            block_unit_size=self.block_unit_size, disk_sector_size=self.disk_sector_size,
                            open_c_reader=open_c_reader, lock=self.lock)

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

    def read_one_element_mmap(self):
        """
        wip
        :return:
        """

        pass

    def __next__(self):
        super().__next__()
        v = self.read_one_req()
        if v is not None:
            return v
        else:
            raise StopIteration
