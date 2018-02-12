# coding=utf-8
"""
    this module provides reader for reading plainText trace

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/07

"""

from PyMimircache.cacheReader.abstractReader import AbstractReader
from PyMimircache.const import ALLOW_C_MIMIRCACHE, INSTALL_PHASE

if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.CacheReader as c_cacheReader


class PlainReader(AbstractReader):
    """
    PlainReader class

    """
    all = ["read_one_req", "copy", "get_params"]

    def __init__(self, file_loc, data_type='c', open_c_reader=True, **kwargs):
        """
        :param file_loc:            location of the file
        :param data_type:           type of data, can be "l" for int/long, "c" for string
        :param open_c_reader:       bool for whether open reader in C backend
        :param kwargs:              not used now
        """

        super(PlainReader, self).__init__(file_loc, data_type, open_c_reader=open_c_reader, lock=kwargs.get("lock"))
        self.trace_file = open(file_loc, 'rb')
        if ALLOW_C_MIMIRCACHE and open_c_reader:
            self.c_reader = c_cacheReader.setup_reader(file_loc, 'p', data_type=data_type, block_unit_size=0)

    def read_one_req(self):
        """
        read one request
        :return: a request
        """
        super().read_one_req()

        line = self.trace_file.readline().decode()
        while line and len(line.strip()) == 0:
            line = self.trace_file.readline().decode()

        if line and len(line.strip()):
            return line.strip()
        else:
            return None

    def read_complete_req(self):
        """
        read all information about one record, which is the same as read_one_req for PlainReader
        """

        return self.read_one_req()

    def skip_n_req(self, n):
        """
        skip N requests from current position

        :param n: the number of requests to skip
        """

        for i in range(n):
            self.read_one_req()


    def copy(self, open_c_reader=False):
        """
        reader a deep copy of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_c_reader: whether open_c_reader_or_not, default not open
        :return: a copied reader
        """

        return PlainReader(self.file_loc, data_type=self.data_type, open_c_reader=open_c_reader, lock=self.lock)

    def get_params(self):
        """
        return all the parameters for this reader instance in a dictionary
        :return: a dictionary containing all parameters
        """

        return {
            "file_loc": self.file_loc,
            "data_type": self.data_type,
            "open_c_reader": self.open_c_reader
        }

    def __next__(self):  # Python 3
        super().__next__()
        element = self.trace_file.readline().strip()
        if element:
            return element
        else:
            raise StopIteration

    def __repr__(self):
        return "PlainReader of trace {}".format(self.file_loc)
