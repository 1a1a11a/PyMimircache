# coding=utf-8

"""
    binaryReader for reading binary trace

    created 2016/06
    refactored 2019/12

    Author: Jason Yang <peter.waynechina@gmail.com>

"""

# import os, sys
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import io
import struct
from PyMimircache.cacheReader.abstractReader import AbstractReader
from PyMimircache.utils.printing import DEBUG, INFO, WARNING, ERROR
from PyMimircache.cacheReader.request import Request, FullRequest

class BinaryReader(AbstractReader):
    """
    BinaryReader class for reading binary trace
    """

    def __init__(self, trace_path, obj_id_type, init_params,
                 open_libm_reader=True, *args, **kwargs):
        """
        initialization function for binaryReader

        the init_params specify the parameters for opening the trace, it is a dictionary of following key-item pairs

        +------------------+--------------+---------------------+---------------------------------------------------+
        | Keyword Argument | Value Type   | Default Value       | Description                                       |
        +==================+==============+=====================+===================================================+
        | obj_id_field     | int          | this is required    | the column of the obj_id                          |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | fmt_field        | string       | this is required    | fmt string of binary data, same as python struct  |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | real_time_field  | int          |        NA           | the column/field of real time                     |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | op_field         | int          |        NA           | the column/field of operation (read/write)        |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | obj_size_field   | int          |        NA           | the column/field of object/block  size            |
        +------------------+--------------+---------------------+---------------------------------------------------+
        | cnt_field        | int          |        NA           | the column/field of request count                 |
        +------------------+--------------+---------------------+---------------------------------------------------+


        :param trace_path:            location of the file
        :param init_params:         init_params for binaryReader, see above
        :param obj_id_type:           type of obj_id, can be "l" for int/long, "c" for string,
                                        if the obj_id is a 64-bit number (on 64-bit system),
                                        recommend using "l" for better performance
        :param open_libm_reader:       a bool variable for whether opening reader in libMimircache
        :param args:              not used for now
        :param kwargs:              not used for now
        """

        super(BinaryReader, self).__init__(trace_path, "binary", obj_id_type, init_params,
                                           open_libm_reader, *args, **kwargs)
        assert 'fmt' in init_params, "please provide format string(fmt) in init_params"

        self.fmt = init_params['fmt']
        self.struct_ins = struct.Struct(self.fmt)
        self.record_size = struct.calcsize(self.fmt)
        assert self.file_size % self.record_size == 0, \
            "file size ({}) is not multiple of record size ({})".format(self.file_size, self.record_size)
        self.num_of_req = self.file_size // self.record_size

        # this might cause problem on Windows and it does give any performance gain
        # self.mm = mmap.mmap(self.trace_file.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
        # self.trace_file_original = self.trace_file
        # self.trace_file = self.mm

    def get_num_of_req(self):
        """
        count the number of requests in the trace,
        just use size to calculate in binary trace,
        so it is very fast
        :return: the number of requests in the trace
        """

        if self.num_of_req <= 0:
            self.num_of_req = self.file_size // self.record_size
        return self.num_of_req

    def read_last_req(self):
        """
        read the last request, binary trace has an efficient implementation

        :return: a request
        """

        self.trace_file.seek(-self.record_size, 2)
        req = self.read_one_req()
        self.reset()
        return req

    def skip_n_req(self, n):
        """
        skip n requests from current position

        :param n: the number of requests to skip
        """

        self.trace_file.seek(self.record_size * n, io.SEEK_CUR)

    def read_one_req(self):
        """
        read one request
        :return: a request
        """

        AbstractReader.read_one_req(self)
        try:
            b = self.trace_file.read(self.record_size)
            if not b:
                return None
            item = self.struct_ins.unpack(b)
            obj_id = item[self.obj_id_field]

            obj_size = item[self.obj_size_field] if self.obj_size_field != -1 else 1
            real_time = item[self.real_time_field] if self.real_time_field != -1 else None
            cnt = item[self.cnt_field] if self.cnt_field != -1 else None
            op = item[self.op_field] if self.op_field != -1 else None

            req = Request(logical_time=self.n_read_req,
                          real_time=real_time, obj_id=obj_id,
                          obj_size=obj_size,
                          cnt=cnt, op=op)
        except Exception as e:
            ERROR("BinaryReader err: {}".format(e))
            self.n_read_req -= 1
            return self.read_one_req()
        return req

    def clone(self, open_libm_reader=False):
        """
        reader a deep clone of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_libm_reader: whether open libm_reader, default not
        :return: a cloned reader
        """

        return BinaryReader(self.trace_path, self.obj_id_type, self.init_params,
                            open_libm_reader=open_libm_reader, lock=self.lock)

    def get_params(self):
        """
        return all the parameters for this reader instance in a dictionary
        :return: a dictionary containing all parameters
        """

        return {
            "trace_path": self.trace_path,
            "trace_type": "binary",
            "init_params": self.init_params,
            "obj_id_type": self.obj_id_type,
            "open_libm_reader": self.open_libm_reader,
            "lock": self.lock
        }


