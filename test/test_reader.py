# coding=utf-8
"""
this module tests CCacheReader and PyCacheReader

Author: Jason Yang <peter.waynechina@gmail.com> 2016/08

"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))

import unittest
import PyMimircache.CMimircache.CacheReader as c_cacheReader
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader

DAT_FOLDER = "../data/"
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"


class CReaderTest(unittest.TestCase):
    def test_c_reader_vscsi(self):
        reader = c_cacheReader.setup_reader("{}/trace.vscsi".format(DAT_FOLDER), 'v')
        lines = c_cacheReader.get_num_of_req(reader)
        self.assertEqual(lines, 113872)

        first_request = c_cacheReader.read_one_req(reader)
        self.assertEqual(int(first_request), 42932745)  # +1 is to avoid block 0

        c_cacheReader.reset_reader(reader)
        first_request = c_cacheReader.read_one_req(reader)
        self.assertEqual(int(first_request), 42932745)
        c_cacheReader.close_reader(reader)

    def test_c_reader_plain(self):
        reader = c_cacheReader.setup_reader("{}/trace.txt".format(DAT_FOLDER), 'p')
        lines = c_cacheReader.get_num_of_req(reader)
        self.assertEqual(lines, 113872)
        first_request = c_cacheReader.read_one_req(reader)
        self.assertEqual(int(first_request), 42932745)

        c_cacheReader.reset_reader(reader)
        first_request = c_cacheReader.read_one_req(reader)
        self.assertEqual(int(first_request), 42932745)
        c_cacheReader.close_reader(reader)

    def test_c_reader_csv(self):
        reader = c_cacheReader.setup_reader("{}/trace.csv".format(DAT_FOLDER), 'c', data_type='c',
                                            init_params={"header": True, "delimiter": ",", "label": 5, "size": 4})
        lines = c_cacheReader.get_num_of_req(reader)
        self.assertEqual(lines, 113872)
        first_request = c_cacheReader.read_one_req(reader)
        self.assertEqual(first_request, "42932745")

        c_cacheReader.reset_reader(reader)
        first_request = c_cacheReader.read_one_req(reader)
        self.assertEqual(first_request, "42932745")
        c_cacheReader.close_reader(reader)

    def test_c_reader_binary(self):
        reader = c_cacheReader.setup_reader("{}/trace.vscsi".format(DAT_FOLDER), 'b', data_type='l',
                                            init_params={"label": 6, "real_time": 7, "fmt": "<3I2H2Q"})
        lines = c_cacheReader.get_num_of_req(reader)
        self.assertEqual(lines, 113872)
        first_request = c_cacheReader.read_one_req(reader)
        self.assertEqual(int(first_request), 42932745)

        c_cacheReader.reset_reader(reader)
        first_request = c_cacheReader.read_one_req(reader)
        self.assertEqual(int(first_request), 42932745)
        c_cacheReader.close_reader(reader)

    def test_c_reader_potpourri(self):
        v_reader = c_cacheReader.setup_reader("{}/trace.vscsi".format(DAT_FOLDER), 'v')
        c_reader = c_cacheReader.setup_reader("{}/trace.csv".format(DAT_FOLDER), 'c', data_type='l',
                                             init_params={"header": True, "delimiter": ",", "label": 5, "size": 4})
        e1 = c_cacheReader.read_one_req(v_reader)
        e2 = c_cacheReader.read_one_req(c_reader)
        while e1 and e2:
            self.assertEqual(e1, e2)
            e1 = c_cacheReader.read_one_req(v_reader)
            e2 = c_cacheReader.read_one_req(c_reader)


class PyReaderTest(unittest.TestCase):
    def test_reader_v(self):
        reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        self.assertEqual(reader.get_num_of_req(), 113872)
        reader.reset()
        lines = 0
        for _ in reader:
            lines += 1
        self.assertEqual(lines, 113872)
        reader.reset()

        # verify read content
        first_request = reader.read_one_req()
        self.assertEqual(int(first_request), 42932745)

        t, req = reader.read_time_req()
        self.assertAlmostEqual(t, 5633898611441.0)
        self.assertEqual(req, 42932746)

    def test_reader_binary(self):
        reader = BinaryReader("{}/trace.vscsi".format(DAT_FOLDER), data_type='l',
                              init_params={"label": 6, "real_time": 7, "fmt": "<3I2H2Q"})
        self.assertEqual(reader.get_num_of_req(), 113872)
        reader.reset()
        lines = 0
        for _ in reader:
            lines += 1
        self.assertEqual(lines, 113872)
        reader.reset()

        # verify read content
        first_request = reader.read_one_req()
        self.assertEqual(int(first_request), 42932745)

        t, req = reader.read_time_req()
        self.assertAlmostEqual(t, 5633898611441.0)
        self.assertEqual(req, 42932746)

        line = reader.read_complete_req()
        self.assertListEqual(line, [2147483880, 512, 1, 42, 256, 42932747, 5633898745540])

    def test_reader_csv(self):
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header": True, "real_time": 2, "op": 3, "size": 4, 'label': 5,
                                        'delimiter': ','})
        self.assertEqual(reader.get_num_of_req(), 113872)
        reader.reset()
        lines = 0
        for _ in reader:
            lines += 1
        self.assertEqual(lines, 113872)
        reader.reset()

        # verify read content
        first_request = reader.read_one_req()
        self.assertEqual(first_request, "42932745")

        t, req = reader.read_time_req()
        self.assertAlmostEqual(t, 5633898611441.0)
        self.assertEqual(req, "42932746")

        line = reader.read_complete_req()
        self.assertListEqual(line, ['1', '5633898745540', '2a', '512', '42932747'])

    def test_reader_csv_datatype_l(self):
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER), data_type="l",
                           init_params={"header": True, "real_time": 2, "op": 3, "size": 4, 'label': 5,
                                        'delimiter': ','})
        self.assertEqual(reader.get_num_of_req(), 113872)
        reader.reset()
        lines = 0
        for _ in reader:
            lines += 1
        self.assertEqual(lines, 113872)
        reader.reset()

        # verify read content
        first_request = reader.read_one_req()
        self.assertEqual(first_request, 42932745)

        t, req = reader.read_time_req()
        self.assertAlmostEqual(t, 5633898611441.0)
        self.assertEqual(req, 42932746)

        line = reader.read_complete_req()
        self.assertListEqual(line, ['1', '5633898745540', '2a', '512', '42932747'])

    def test_reader_plain(self):
        reader = PlainReader("{}/trace.txt".format(DAT_FOLDER))
        self.assertEqual(reader.get_num_of_req(), 113872)
        reader.reset()
        lines = 0
        for _ in reader:
            lines += 1
        self.assertEqual(lines, 113872)
        reader.reset()

        # verify read content
        first_request = reader.read_one_req()
        self.assertEqual(first_request, "42932745")

    def test_reader_potpourri(self):
        v_reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        c_reader = CsvReader("{}/trace.csv".format(DAT_FOLDER), data_type="l",
                             init_params={"header": True, "real_time": 2, "op": 3, "size": 4, 'label': 5,
                                          'delimiter': ','})

        for req1, req2 in zip(v_reader, c_reader):
            self.assertEqual(req1, req2)

    def test_context_manager(self):
        with VscsiReader("{}/trace.vscsi".format(DAT_FOLDER)) as reader:
            self.assertEqual(reader.get_num_of_req(), 113872)


if __name__ == "__main__":
    unittest.main()
