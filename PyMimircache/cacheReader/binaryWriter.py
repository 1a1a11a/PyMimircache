# coding=utf-8
"""
    this module provides a binary writer, which can be used for generating binary traces
    or converting other types of traces into binary traces

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/06

"""

import struct


class BinaryWriter:
    """
    class for writing binary traces
    """
    def __init__(self, ofilename, fmt="",
                 including_fields=("real_time", "obj_id", "obj_size"),
                 *args, **kwargs):
        """
        initialize a binary trace writer

        :param ofilename: output file name
        :param fmt: format of binary trace
        """

        self.ofilename = ofilename
        self.ofile = open(ofilename, 'wb')
        self.fmt = fmt
        self.including_fields = including_fields

        if len(self.fmt) == 0:
            for field in including_fields:
                if field in ("real_time", "obj_id", "obj_size"):
                    self.fmt += "I"
                else:
                    raise RuntimeError("I don't know how to support this, please provide fmt instead")

        self.struct_ins = struct.Struct(self.fmt)

    def convert_trace(self, reader):
        """
        convert a given trace to a binary trace
        :param reader:
        :return:
        """

        obj_id_map = {}
        for req in reader:
            item = []
            for field in self.including_fields:
                field_content = getattr(req, field)
                if field == "obj_id":
                    field_content = obj_id_map.get(field_content, len(obj_id_map)+1)
                item.append(field_content)
            b = self.struct_ins.pack(*item)
            self.ofile.write(b)

    def write(self, item):
        """
        write value into binary file

        :param item: binary value

        """
        assert isinstance(item, tuple)
        b = self.struct_ins.pack(*item)
        self.ofile.write(b)

    def close(self):
        """
        close writer

        """
        if getattr(self, "ofile", None):
            self.ofile.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

