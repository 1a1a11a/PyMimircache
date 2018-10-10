# coding=utf-8

"""
    this module provides a binary writer, which can be used for generating binary traces
    or converting other types of traces into binary traces

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/06

"""

import struct


class TraceBinaryWriter:
    """
    class for writing binary traces
    """
    def __init__(self, ofilename, fmt):
        """
        initialize a binary trace writer

        :param ofilename: output file name
        :param fmt: format of binary trace
        """

        self.ofilename = ofilename
        self.ofile = None
        if self.ofile is None:
            try:
                self.ofile = open(ofilename, 'wb')
            except Exception as e:
                raise RuntimeError("failed to create output file {}, {}".format(ofilename, e))
        self.fmt = fmt
        self.structIns = struct.Struct(self.fmt)


    def write(self, value):
        """
        write value into binary file

        :param value: binary value

        """
        assert isinstance(value, tuple)
        b = self.structIns.pack(*value)
        self.ofile.write(b)


    def close(self):
        """
        close writer

        """
        self.ofile.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        if self.ofile is None:
            self.ofile = open(self.ofilename, 'wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

