# coding=utf-8

"""
read a binary format trace
"""

import io, os, struct
import mimircache.c_cacheReader as c_cacheReader
from mimircache.cacheReader.abstractReader import cacheReaderAbstract


class binaryReader(cacheReaderAbstract):
    def __init__(self, file_loc, init_params, data_type='c', block_unit_size=0, disk_sector_size=0, open_c_reader=True):
        """
        initialization function
        :param file_loc:
        :param init_params:
        :param data_type:
        :param open_c_reader:
        """
        super(binaryReader, self).__init__(file_loc, data_type, block_unit_size, disk_sector_size)
        self.file_loc = file_loc
        assert os.path.exists(file_loc), "provided data file does not exist"
        assert 'fmt' in init_params, "please provide format string(fmt) in init_params"
        assert "label" in init_params, "please specify the order of label, beginning from 1"
        if block_unit_size != 0:
            assert "size" in init_params, "please provide size option to consider request size"

        self.fmt = init_params['fmt']
        self.label_column = init_params['label']
        self.time_column = init_params.get("real_time", -1)
        self.trace_file = open(file_loc, 'rb')
        self.structIns = struct.Struct(self.fmt)
        self.record_size = struct.calcsize(self.fmt)
        self.data_type = data_type
        self.init_params = init_params

        if open_c_reader:
            # the data type here is not real data type, it will auto correct in C
            self.cReader = c_cacheReader.setup_reader(file_loc, 'b', data_type=self.data_type,
                                                      block_unit_size=block_unit_size,
                                                      disk_sector_size=disk_sector_size, 
                                                      init_params=init_params)



    def read_one_element(self):
        """
        read one request, only return the label of the request
        :return:
        """
        super().read_one_element()
        b = self.trace_file.read(self.record_size)
        if b and len(b):
            ret = self.structIns.unpack(b)[self.label_column -1]
            if self.data_type == 'l':
                if ret and self.block_unit_size != 0 and self.disk_sector_size != 0:
                    ret = int(ret) * self.disk_sector_size // self.block_unit_size
                return ret
            else:
                return ret
        else:
            return None


    def read_whole_line(self):
        """
        read the complete line, including request and its all related info
        :return:
        """
        super().read_one_element()
        b = self.trace_file.read(self.record_size)
        if len(b):
            ret = list(self.structIns.unpack(b))
            if self.block_unit_size != 0 and self.disk_sector_size != 0:
                ret[self.label_column-1] = ret[self.label_column-1] * self.disk_sector_size // self.block_unit_size
            return ret
        else:
            return None


    def lines(self):
        b = self.trace_file.read(self.record_size)
        while len(b):
            ret = list(self.structIns.unpack(b))
            if self.block_unit_size != 0 and self.disk_sector_size != 0:
                ret[self.label_column-1] = ret[self.label_column-1] * self.disk_sector_size // self.block_unit_size
            b = self.trace_file.read(self.record_size)
            yield ret


    def read_time_request(self):
        """
        return real_time information for the request in the form of (time, request)
        :return:
        """
        assert self.time_column!=-1, "you need to provide time in order to use this function"
        super().read_one_element()
        b = self.trace_file.read(self.record_size)
        if len(b):
            ret = self.structIns.unpack(b)
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


    def jumpN(self, n):
        """
        jump N records from current position
        :return:
        """
        self.trace_file.seek(struct.calcsize(self.fmt) * n, io.SEEK_CUR)


    def __next__(self):
        super().__next__()
        v = self.read_one_element()
        if v is not None:
            return v
        else:
            raise StopIteration
