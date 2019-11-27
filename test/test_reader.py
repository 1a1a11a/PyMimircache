# coding=utf-8
"""
this module tests CCacheReader and PyCacheReader

Author: Jason Yang <peter.waynechina@gmail.com> 2016/08

"""

import os
import sys

import unittest
from PyMimircache import *
import libMimircache.PyUtils as PyUtils


DAT_FOLDER = "../data/"
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"


class libMimircacheReaderTest(unittest.TestCase):
    def test_c_reader_vscsi(self):
        reader = PyUtils.setup_reader(trace_path="{}/trace.vscsi".format(DAT_FOLDER), trace_type='v', obj_id_type="l")
        PyUtils.reset_reader(reader)
        reader = PyUtils.setup_reader(f"{DAT_FOLDER}/trace.csv", 'p', obj_id_type="l")
        PyUtils.reset_reader(reader)
        reader = PyUtils.setup_reader(f"{DAT_FOLDER}/trace.csv", 'p', obj_id_type="c")
        PyUtils.reset_reader(reader)
        reader = PyUtils.setup_reader(f"{DAT_FOLDER}/trace.csv", 'c', obj_id_type='c',
                                      init_params={"has_header": True, "obj_id_field": 5, "obj_size_field": 4,
                                                   "real_time_field": 2})
        PyUtils.reset_reader(reader)
        reader = PyUtils.setup_reader(f"{DAT_FOLDER}/trace.csv", 'c', obj_id_type='l',
                                      init_params={"has_header": True, "obj_id_field": 5, "obj_size_field": 4,
                                                   "real_time_field": 2})
        PyUtils.reset_reader(reader)
        reader = PyUtils.setup_reader(f"{DAT_FOLDER}/trace.vscsi", 'b', obj_id_type='l',
                                      init_params={"obj_id_field": 6, "real_time_field": 7, "fmt": "<3I2H2Q", "obj_size_field": 2})
        PyUtils.reset_reader(reader)


class PyMimircacheReaderTest(unittest.TestCase):
    def verify_reader(self, reader):
        self.assertEqual(reader.get_num_of_req(), 113872)
        reader.reset()

        # verify trace content
        req = next(reader)
        self.assertEqual(int(req.obj_id), 42932745)

        req = reader.read_one_req()
        self.assertEqual(req.logical_time, 2)
        self.assertEqual(int(req.obj_id), 42932746)
        if reader.trace_type[0] == "p":
            self.assertEqual(req.obj_size, 1)
        else:
            self.assertEqual(req.real_time, 5633898611441)
            self.assertEqual(req.obj_size, 512)
        if reader.obj_id_type == 'c':
            self.assertEqual(req.obj_id, "42932746")

        # check request cnt and last req
        n_req = 2
        for req in reader:
            n_req += 1
        self.assertEqual(n_req, 113872)
        self.assertEqual(req.logical_time, 113872)
        self.assertEqual(int(req.obj_id), 42936150)
        if reader.trace_type[0] == "p":
            self.assertEqual(req.obj_size, 1)
        else:
            self.assertEqual(req.real_time, 5641098458687)
            self.assertEqual(req.obj_size, 512)
        reader.reset()

    def test_all_readers(self):
        readers = [
            # VscsiReader("{}/trace.vscsi".format(DAT_FOLDER)),
            PlaintxtReader("{}/trace.txt".format(DAT_FOLDER), obj_id_type='c'),
            PlaintxtReader("{}/trace.txt".format(DAT_FOLDER), obj_id_type='l'),
            BinaryReader("{}/trace.vscsi".format(DAT_FOLDER), obj_id_type='l',
                         init_params={"obj_id_field": 6, "real_time_field": 7, "obj_size_field": 2, "fmt": "<3I2H2Q"}),
            CsvReader("{}/trace.csv".format(DAT_FOLDER), obj_id_type="l",
                      init_params={"has_header": True, "real_time_field": 2,
                                   "op_field": 3, "obj_size_field": 4, 'obj_id_field': 5}),
            CsvReader("{}/trace.csv".format(DAT_FOLDER), obj_id_type="c",
                      init_params={"has_header": True, "real_time_field": 2,
                                   "op_field": 3, "obj_size_field": 4, 'obj_id_field': 5}),
        ]
        for reader in readers:
            self.verify_reader(reader)
            print("done")

    def test_context_manager(self):
        with VscsiReader("{}/trace.vscsi".format(DAT_FOLDER)) as reader:
            self.assertEqual(reader.get_num_of_req(), 113872)



if __name__ == "__main__":
    unittest.main()
