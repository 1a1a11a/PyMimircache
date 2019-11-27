# coding=utf-8
"""
    vscsi reader for vscsi trace

    created 2016/06
    refactored 2019/12

    Author: Jason Yang <peter.waynechina@gmail.com>

"""

from PyMimircache.cacheReader.binaryReader import BinaryReader


class VscsiReader(BinaryReader):
    """
    VscsiReader for vscsi trace
     
    """

    def __init__(self, trace_path, vscsi_type=1, open_libm_reader=True, *args, **kwargs):
        """
        :param trace_path:            location of the file
        :param vscsi_type:          vscsi trace type, can be 1 or 2
        :param open_libm_reader:       bool for whether open reader in C backend
        """

        self.vscsi_type = vscsi_type
        if vscsi_type == 1:
            init_params = {"obj_size_field": 2, "op_field": 4, "obj_id_field": 6, "real_time_field": 7,
                           "fmt": "<3I2H2Q"}
            assert "vscsi2" not in trace_path, "are you sure the trace ({}) is vscsi type 1? ".format(trace_path)
        elif vscsi_type == 2:
            init_params = {"op_field": 1, "obj_size_field": 4, "obj_id_field": 6, "real_time_field": 7,
                           "fmt": "<2H3I3Q"}
            assert "vscsi1" not in trace_path, "are you sure the trace ({}) is vscsi type 2? ".format(trace_path)
        else:
            raise RuntimeError("unknown vscsi type")

        super(VscsiReader, self).__init__(trace_path, 'l', init_params,
                                          open_libm_reader,
                                          *args, **kwargs)

    def clone(self, open_libm_reader=False):
        """
        reader a deep clone of current reader with everything reset to initial state,
        the returned reader should not interfere with current reader

        :param open_libm_reader: whether open libm_reader, default not
        :return: a cloned reader
        """

        return VscsiReader(self.trace_path, self.vscsi_type, open_libm_reader, lock=self.lock)

    def get_params(self):
        """
        return all the parameters for this reader instance in a dictionary
        :return: a dictionary containing all parameters
        """

        return {
            "trace_path": self.trace_path,
            "trace_type": "vscsi",
            "vscsi_type": self.vscsi_type,
            "open_libm_reader": self.open_libm_reader,
            "lock": self.lock
        }

