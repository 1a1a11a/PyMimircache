# # coding=utf-8
# # coding=utf-8
#
#
# import os
# import sys
# sys.path.append(os.path.join(os.getcwd(), "../.."))
# import unittest
#
# from PyMimircache.cacheReader.csvReader import CsvReader
# from PyMimircache.cacheReader.plainReader import PlainReader
# from PyMimircache.cacheReader.vscsiReader import VscsiReader
# from PyMimircache.cacheReader.binaryReader import BinaryReader
# from PyMimircache.profiler.cGeneralProfiler import CGeneralProfiler
# from PyMimircache.profiler.pyGeneralProfiler import PyGeneralProfiler
#
#
# DAT_FOLDER = "../data/"
# if not os.path.exists(DAT_FOLDER):
#     if os.path.exists("../../data/"):
#         DAT_FOLDER = "../../data/"
#     elif os.path.exists("../../PyMimircache/data/"):
#         DAT_FOLDER = "../../PyMimircache/data/"
#     elif os.path.exists("../PyMimircache/data/"):
#         DAT_FOLDER = "../PyMimircache/data/"
#     else:
#         raise RuntimeError("cannot find data")
#
# class optimalAlgTest(unittest.TestCase):
#
#
#     def t2est_Optimal_datatype_l(self):
#         reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
#         p = CGeneralProfiler(reader, "Optimal", cache_size=2000, num_of_threads=os.cpu_count())
#         p2 = PyGeneralProfiler(reader, 'Optimal', cache_size=2000, num_of_threads=os.cpu_count())
#
#         hc = p.get_hit_count()
#         hc2 = p2.get_hit_count()
#         self.assertEqual(hc[0], 0)
#         self.assertEqual(hc2[0], 0)
#         self.assertEqual(hc[10], 180)
#         self.assertEqual(hc2[10], 180)
#         self.assertListEqual(hc, hc2)
#
#
#         hr = p.get_hit_ratio()
#         hr2 = p2.get_hit_ratio()
#         self.assertAlmostEqual(hr[100], 0.28106996417045593)
#         self.assertAlmostEqual(hr2[100], 0.28106996417045593)
#
#         p.plotHRC(figname="cGeneralProfiler_Optimal_l.png")
#         p2.plotHRC(figname="pyGeneralProfiler_Optimal_l.png")
#
#         reader.close()
#
#
#     def test_Optimal_datatype_c(self):
#         reader = CsvReader("{}/trace.csv".format(DAT_FOLDER), data_type="c",
#                            init_params={"header":True, 'label':5, 'delimiter':','})
#
#         py_reader = reader.copy(open_c_reader=False)
#
#         p = CGeneralProfiler(reader, "Optimal", cache_size=2000, cache_params={"reader":py_reader}, num_of_threads=os.cpu_count())
#         p2 = PyGeneralProfiler(reader, 'Optimal', cache_size=2000, cache_params={"reader":py_reader}, num_of_threads=os.cpu_count())
#
#         hc = p.get_hit_count()
#         hc2 = p2.get_hit_count()
#         self.assertEqual(hc[0], 0)
#         self.assertEqual(hc2[0], 0)
#         self.assertEqual(hc[10], 180)
#         self.assertEqual(hc2[10], 180)
#         self.assertListEqual(hc, hc2)
#
#
#         hr = p.get_hit_ratio()
#         hr2 = p2.get_hit_ratio()
#         self.assertAlmostEqual(hr[100], 0.28106996417045593)
#         self.assertAlmostEqual(hr2[100], 0.28106996417045593)
#
#         p.plotHRC(figname="cGeneralProfiler_Optimal_c.png")
#         p2.plotHRC(figname="pyGeneralProfiler_Optimal_c.png")
#
#         reader.close()
#
#
# if __name__ == "__main__":
#     unittest.main()
