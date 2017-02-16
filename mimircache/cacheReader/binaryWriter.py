# coding=utf-8

"""
save as a binary format trace
"""

import struct


class traceBinaryWriter:
    def __init__(self, ofilename, fmt):
        self.ofilename = ofilename
        self.ofile = None
        if self.ofile is None:
            self.ofile = open(ofilename, 'wb')
        self.fmt = fmt


    def write(self, value):
        b = struct.pack(self.fmt, *value)
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

