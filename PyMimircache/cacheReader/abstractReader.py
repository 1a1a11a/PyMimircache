# coding=utf-8
"""
    reader interface

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/06

"""

from abc import ABC, abstractmethod
import os
from collections import defaultdict
from PyMimircache.const import IMPORT_LIBMIMIRCACHE
from multiprocessing import Manager

if IMPORT_LIBMIMIRCACHE:
    import libMimircache.PyUtils as PyUtils


class AbstractReader(ABC):
    """
    reader interface

    """

    def __init__(self, trace_path, trace_type, obj_id_type,
                 init_params=None,
                 open_libm_reader=False, lock=None, *args, **kwargs):
        """
            the initialization abstract function for cacheReaderAbstract

        :param trace_path:          location of the file
        :param trace_type:          type of the trace
        :param obj_id_type:         type of obj_id, can be "l" for int/long, "c" for string
        :param init_params:            init_params for reader
        :param open_libm_reader:    whether open libMimircache reader
        :param lock:
        :param args:
        :param kwargs:
        """

        self.trace_path = trace_path
        self.trace_type = trace_type
        self.obj_id_type = obj_id_type
        self.open_libm_reader = open_libm_reader
        self.init_params = init_params

        self.real_time_field = -1
        self.obj_id_field = -1
        self.obj_size_field = -1
        self.cnt_field = -1
        self.op_field = -1

        if init_params:
            assert "obj_id_field" in init_params, "please specify the field/column of obj_id, beginning from 1"

            # this index begins from 1, so need to reduce by one before use
            self.obj_id_field = init_params.get("obj_id_field", 0) - 1
            self.real_time_field = init_params.get("real_time_field", 0) - 1
            self.obj_size_field = init_params.get("obj_size_field", 0) - 1
            self.cnt_field = init_params.get("cnt_field", 0) - 1
            self.op_field = init_params.get("op_field", 0) - 1

        self.trace_file = None
        self.libm_reader = None

        assert (os.path.exists(trace_path)), "trace file({}) does not exist".format(trace_path)
        assert len(args) == 0, "args not empty {}".format(args)
        assert len(kwargs) == 0, "kwargs not empty {}".format(kwargs)

        self.lock = lock
        if self.lock is None:
            self._mp_manager = Manager()
            self.lock = self._mp_manager.Lock()

        self.n_read_req = 0
        self.num_of_req = -1
        self.num_of_obj = -1
        self.file_size = 0

        self.support_real_time = True if self.real_time_field != -1 else False
        self.support_size = True if self.obj_size_field != -1 else False
        self.already_load_rd = False
        self._open_trace()

    def _open_trace(self):
        if self.trace_file:
            return
        if self.trace_type.lower() == "binary" or self.trace_type.lower() == "b" or self.trace_type.lower() == "bin":
            self.trace_file = open(self.trace_path, "rb")
        else:
            self.trace_file = open(self.trace_path, encoding='utf-8', errors='ignore')

        # self.trace_file = open(self.trace_path, "rb")
        if self.open_libm_reader:
            self.libm_reader = PyUtils.setup_reader(self.trace_path, self.trace_type, self.obj_id_type,
                                                    self.init_params)

        self.file_size = os.path.getsize(self.trace_path)
        self.n_read_req = 0

    def close(self):
        """
            close reader and free all resources used by reader,
            this is especially important for libm_reader
        """
        try:
            if self is not None:
                if getattr(self, "trace_file", None):
                    self.trace_file.close()
                    self.trace_file = None
                # libMimircache reader will close upon gc
                # if getattr(self, "libm_reader", None) and globals().get("PyUtils", None) is not None:
                #     PyUtils.close_reader(self.libm_reader)
                #     self.libm_reader = None
                #     print("close libm_reader")
        except Exception as e:
            print("Exception during close reader: {}".format(e))

    def reset(self):
        """
        reset the read location back to beginning, similar as rewind in POSIX
        """
        self.n_read_req = 0
        self.trace_file.seek(0, 0)
        if self.open_libm_reader:
            PyUtils.reset_reader(self.libm_reader)

    def get_num_of_req(self):
        """
        count the number of requests in the trace, fast for binary type trace,
        for plain/csv type trace, this is slow
        :return: the number of requests in the trace
        """

        if self.num_of_req <= 0:
            # clear before counting
            self.num_of_req = 0
            while self.read_one_req():
                self.num_of_req += 1
            self.reset()

        return self.num_of_req

    def get_num_of_obj(self):
        """
        count the number of unique block/obj in the trace
        :return: the number of unique block/obj
        """

        if self.num_of_obj == -1:
            self.num_of_obj = len(self.get_obj_freq_dict())
        return self.num_of_obj

    def get_obj_freq_dict(self):
        """
        calculate the count (frequency) for each block/obj
        :return: a dictionary mapping from block/obj to count
        """

        d = defaultdict(int)
        for req in self:
            d[req.obj_id] += 1
        self.reset()
        return d

    def read_first_req(self):
        """
        read the first request
        :return: the first request
        """

        self.reset()
        return self.read_one_req()

    def read_last_req(self):
        """
        read the last request
        :return: the last request
        """

        req = self.read_one_req()
        last_req = req
        while req:
            last_req = req
            req = self.read_one_req()

        self.reset()
        return last_req

    def skip_n_req(self, n):
        """
        an efficient way to skip N requests from current position

        :param n: the number of requests to skip
        """

        for i in range(n):
            self.read_one_req()

    def get_avg_req_size(self):
        """
            calculates the avg request size
        :return: the avg request size
        """

        size_sum = 0
        req_cnt = 0

        for req in self:
            size_sum += req.req_size
            req_cnt += 1
        self.reset()
        return size_sum / req_cnt


    def __iter__(self):
        return self

    def next(self):
        """ part of the iterator implementation """
        return self.__next__()

    def __next__(self):
        req = self.read_one_req()
        if req:
            return req
        else:
            raise StopIteration

    def __len__(self):
        return self.get_num_of_req()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    # @atexit.register
    def __del__(self):
        # print("del called")
        self.close()

    @abstractmethod
    def read_one_req(self):
        """
        read one request from the trace
        :return:
        """

        self.n_read_req += 1
        return None

    @abstractmethod
    def clone(self, open_libm_reader=False):
        """
        reader a deep clone of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_libm_reader: whether open_c_reader_or_not, default not open
        :return: a copied reader
        """
        pass

    @abstractmethod
    def get_params(self):
        """
            return all the parameters for this reader instance in a dictionary
        :return: a dictionary containing all parameters
        """
        pass


