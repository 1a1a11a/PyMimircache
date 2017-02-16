# coding=utf-8

"""
read a binary format trace
"""

import io, os, struct


class traceBinaryReader:
    def __init__(self, infilename, fmt):
        self.infilename = infilename
        self.infile = None
        assert os.path.exists(infilename), "provided data file does not exist"
        if self.infile is None:
            self.infile = open(infilename, 'rb')
        self.fmt = fmt


    def read(self):
        b = self.infile.read(struct.calcsize(self.fmt))
        if len(b):
            return struct.unpack(self.fmt, b)
        else:
            return None

    def reset(self):
        self.infile.seek(0, io.SEEK_SET)

    def jumpN(self, n):
        """
        jump N records from current position
        :return:
        """
        self.infile.seek(struct.calcsize(self.fmt) * n, io.SEEK_CUR)

    def close(self):
        if self.infile:
            self.infile.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        if self.infile is None:
            self.infile = open(self.infilename, 'rb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        v = self.read()
        if v:
            return v
        else:
            raise StopIteration

    def __len__(self):
        return os.path.getsize(self.infilename)//struct.calcsize(self.fmt)


if __name__ == "__main__":
    with traceBinaryReader("../utils/test.vscsi", "<3I2H2Q") as r:
        count = 0
        line = r.read()
        print(line)
        while line:
            count += 1
            line = r.read()
        print(len(r))
    print(count)
