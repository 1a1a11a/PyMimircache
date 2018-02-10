# coding=utf-8
"""
    vscsi reader for vscsi trace

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/06

"""

from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.const import ALLOW_C_MIMIRCACHE, INSTALL_PHASE

if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.CacheReader as c_cacheReader


class VscsiReader(BinaryReader):
    """
    VscsiReader for vscsi trace
     
    """
    all = ["read_one_req", "read_time_req", "read_complete_req",
           "get_average_size", "get_timestamp_list",
           "reset", "copy", "get_params"]

    def __init__(self, file_loc, vscsi_type=1,
                 block_unit_size=0, open_c_reader=True, **kwargs):
        """
        :param file_loc:            location of the file
        :param vscsi_type:          vscsi trace type, can be 1 or 2
        :param block_unit_size:     block size for storage system, 0 when disabled
        :param open_c_reader:       bool for whether open reader in C backend
        """

        if vscsi_type == 1:
            self.vscsi_type = 1
            init_params = {"size": 2, "op": 4, "label": 6, "real_time": 7, "fmt": "<3I2H2Q"}
            assert "vscsi2" not in file_loc, "are you sure the trace ({}) is vscsi type 1? ".format(file_loc)
        elif vscsi_type == 2:
            self.vscsi_type = 2
            init_params = {"op": 1, "size": 4, "label": 6, "real_time": 7, "fmt": "<2H3I3Q"}
            assert "vscsi1" not in file_loc, "are you sure the trace ({}) is vscsi type 2? ".format(file_loc)
        else:
            raise RuntimeError("unknown vscsi type")

        super(VscsiReader, self).__init__(file_loc, data_type='l',
                                          init_params=init_params,
                                          block_unit_size=block_unit_size,
                                          disk_sector_size=512,
                                          open_c_reader=open_c_reader,
                                          lock=kwargs.get("lock", None))

    def get_average_size(self):
        """
        sum sizes for all the requests, then divided by number of requests
        :return: a float of average size of all requests
        """

        sizes = 0
        counter = 0

        t = self.read_complete_req()
        while t:
            sizes += t[2]
            counter += 1
            t = self.read_complete_req()
        self.reset()
        return sizes / counter

    def get_timestamp_list(self):
        """
        get a list of timestamps
        :return: a list of timestamps corresponding to requests
        """

        ts_list = []
        r = c_cacheReader.read_time_req(self.c_reader)
        while r:
            ts_list.append(r[0])
            r = c_cacheReader.read_time_req(self.c_reader)
        return ts_list

    def copy(self, open_c_reader=False):
        """
        reader a deep copy of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_c_reader: whether open_c_reader_or_not, default not open
        :return: a copied reader
        """

        return VscsiReader(self.file_loc, self.vscsi_type, self.block_unit_size, open_c_reader, lock=self.lock)

    def get_params(self):
        """
        return all the parameters for this reader instance in a dictionary
        :return: a dictionary containing all parameters
        """

        return {
            "file_loc": self.file_loc,
            "vscsi_type": self.vscsi_type,
            "block_unit_size": self.block_unit_size,
            "open_c_reader": self.open_c_reader,
            "lock": self.lock
        }

    def __repr__(self):
        return "vscsiReader of trace {}".format(self.file_loc)
