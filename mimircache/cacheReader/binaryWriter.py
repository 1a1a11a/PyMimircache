# coding=utf-8

"""
save as a binary format trace
"""

import struct
from mimircache.utils.printing import *


class traceBinaryWriter:
    def __init__(self, ofilename, fmt):
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
        assert isinstance(value, tuple)
        b = self.structIns.pack(*value)
        self.ofile.write(b)


    def close(self):
        self.ofile.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        if self.ofile is None:
            self.ofile = open(self.ofilename, 'wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

