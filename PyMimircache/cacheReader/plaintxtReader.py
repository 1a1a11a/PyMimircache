# coding=utf-8
"""
    this module provides reader for reading plainText trace

    created 2016/07
    refactored 2019/12

    Author: Jason Yang <peter.waynechina@gmail.com>

"""

from PyMimircache.cacheReader.abstractReader import AbstractReader
from PyMimircache.utils.printing import DEBUG, INFO, WARNING, ERROR
from PyMimircache.cacheReader.request import Request, FullRequest

class PlaintxtReader(AbstractReader):
    """
    reader for reading plain text trace

    """

    def __init__(self, trace_path, obj_id_type='c', open_libm_reader=True, *args, **kwargs):
        """
        :param trace_path:            location of the file
        :param obj_id_type:           type of data, can be "l" for int/long, "c" for string
        :param open_libm_reader:       bool for whether open reader in C backend
        :param kwargs:              not used now
        """

        super(PlaintxtReader, self).__init__(trace_path, "plain", obj_id_type,
                                             open_libm_reader=open_libm_reader, *args, **kwargs)

    def read_one_req(self):
        """
        read one request
        :return: a request
        """
        super().read_one_req()

        line = self.trace_file.readline().strip()
        while line and (len(line) == 0 or line[0] == "#"):      # while line makes sure not a deadloop
            line = self.trace_file.readline().strip()

        if line:
            obj_id = int(line) if self.obj_id_type == 'l' else line
            req = Request(logical_time=self.n_read_req, obj_id=obj_id, obj_size=1)
            return req
        else:
            return None

    def clone(self, open_libm_reader=False):
        """
        reader a deep clone of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_libm_reader: whether open libm_reader, default not
        :return: a cloned reader
        """

        return PlaintxtReader(self.trace_path, obj_id_type=self.obj_id_type,
                              open_libm_reader=open_libm_reader, lock=self.lock)

    def get_params(self):
        """
        return all the parameters for this reader instance in a dictionary
        :return: a dictionary containing all parameters
        """

        return {
            "trace_path": self.trace_path,
            "trace_type": "plain",
            "obj_id_type": self.obj_id_type,
            "open_libm_reader": self.open_libm_reader
        }

