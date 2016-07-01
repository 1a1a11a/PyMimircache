


import unittest
import mimircache.c_cacheReader as c_cacheReader
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader


class cReaderTest(unittest.TestCase):
    def test_reader_v(self):
        reader = c_cacheReader.setup_reader('../mimircache/data/trace.vscsi', 'v')
        lines = c_cacheReader.get_num_of_lines(reader)
        self.assertEqual(lines, 113872)
        first_request = c_cacheReader.read_one_element(reader)
        self.assertEqual(int(first_request), 42932745)
        first_request = None
        c_cacheReader.reset_reader(reader)
        first_request = c_cacheReader.read_one_element(reader)
        self.assertEqual(int(first_request), 42932745)

    def test_reader_p(self):
        reader = c_cacheReader.setup_reader('../mimircache/data/trace.txt', 'p')
        lines = c_cacheReader.get_num_of_lines(reader)
        self.assertEqual(lines, 113872)
        first_request = c_cacheReader.read_one_element(reader)
        self.assertEqual(int(first_request), 42932745)
        first_request = None
        c_cacheReader.reset_reader(reader)
        first_request = c_cacheReader.read_one_element(reader)
        self.assertEqual(int(first_request), 42932745)


class readerTest(unittest.TestCase):
    def test_reader_v(self):
        pass


if __name__ == "__main__":
    unittest.main()

