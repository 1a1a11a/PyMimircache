# coding=utf-8
"""
    a csv trace reader

    created 2016/06
    refactored 2019/12

    Author: Jason Yang <peter.waynechina@gmail.com>

"""

import csv
from PyMimircache.cacheReader.abstractReader import AbstractReader
from PyMimircache.utils.printing import DEBUG, INFO, WARNING, ERROR
from PyMimircache.cacheReader.request import Request, FullRequest

class CsvReader(AbstractReader):
    """
    CsvReader class for reading csv trace
    """

    def __init__(self, trace_path, obj_id_type, init_params,
                 open_libm_reader=True, *args, **kwargs):
        """
        :param trace_path:            location of the file
        :param obj_id_type:           type of obj_id, can be "l" for int/long, "c" for string,
                                        if the obj_id is a 64-bit number (on 64-bit system),
                                        recommend using "l" for better performance
        :param init_params:         the init_params for opening csv
        :param open_libm_reader:       a bool variable for whether opening reader in libMimircache
        :param args:              not used for now
        :param kwargs:              not used for now
        """

        super(CsvReader, self).__init__(trace_path, "csv", obj_id_type, init_params,
                                        open_libm_reader, *args, **kwargs)

        # self.delimiter = init_params.get('delimiter', None)
        # if "delimiter" not in init_params:
        #     INFO("open {} using default delimiter \",\" for CsvReader".format(trace_path))

        self.has_header = init_params.get('has_header', False)
        self.csv_reader = csv.reader(self.trace_file)

        if self.has_header:
            self.header = next(self.csv_reader)


    def read_one_req(self):
        """
        read one request
        :return: a Request
        """
        super().read_one_req()
        item = next(self.csv_reader)
        while item[0][0] == '#':
            item = next(self.csv_reader)
            print("skip {}".format(item))
        try:
            obj_id = int(item[self.obj_id_field]) if self.obj_id_type == 'l' else item[self.obj_id_field]
            obj_size = 1
            if self.obj_size_field != -1 and item[self.obj_size_field] != "":
                obj_size = int(item[self.obj_size_field])

            real_time = int(item[self.real_time_field]) if self.real_time_field != -1 else None
            cnt = int(item[self.cnt_field]) if self.cnt_field != -1 else None
            op = item[self.op_field] if self.op_field != -1 else None

            req = Request(logical_time=self.n_read_req,
                          real_time=real_time, obj_id=obj_id,
                          obj_size=obj_size,
                          cnt=cnt, op=op)

        except Exception as e:
            print("CsvReader err {} at line {}".format(e, item))
            self.n_read_req -= 1
            return self.read_one_req()
        return req

    def reset(self):
        """
        reset reader to initial state
        :return:
        """

        super().reset()
        if self.has_header:
            self.header = next(self.csv_reader)

    def clone(self, open_libm_reader=False):
        """
        reader a deep clone of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_libm_reader: whether open libm_reader, default not
        :return: a cloned reader
        """

        return CsvReader(self.trace_path, self.obj_id_type, self.init_params,
                         open_libm_reader=open_libm_reader, lock=self.lock)

    def get_params(self):
        """
        return all the parameters for this reader instance in a dictionary
        :return: a dictionary containing all parameters
        """

        return {
            "trace_path": self.trace_path,
            "trace_type": "csv",
            "init_params": self.init_params,
            "obj_id_type": self.obj_id_type,
            "open_libm_reader": self.open_libm_reader,
            "lock": self.lock
        }

