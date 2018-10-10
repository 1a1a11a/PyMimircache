"""
this file combines the splitted Akamai datacenter csv file into a bigger trace file, 
meanwhile convert the csv file to binary 
"""


import heapq

from PyMimircache import *
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter

DIR = "./"



for folder in os.listdir(DIR):
    if os.path.isdir("{}/{}".format(DIR, folder)):
        if folder == 'binary':
            continue
        print("current folder {}".format(folder))
        if not os.path.exists("{}/binary/{}/".format(DIR, folder)):
            os.makedirs("{}/binary/{}/".format(DIR, folder))
        else:
            continue
        writer_complete = TraceBinaryWriter(fmt="<LL", ofilename="{}/binary/{}/complete".format(DIR, folder))
        dict_complete = {}
        readers = []
        writers = []
        dicts   = []
        heap    = []
        for f in os.listdir("{}/{}".format(DIR, folder)):
            if not os.path.isfile("{}/{}/{}".format(DIR, folder, f)):
                continue
            readers.append(CsvReader("{}/{}/{}".format(DIR, folder, f),
                                     init_params={"label_column": 5, "real_time_column": 1, "delimiter": "\t"}))
            writers.append(TraceBinaryWriter("{}/binary/{}/{}".format(DIR, folder, f), fmt="<LL"))
            dicts.append({})
            r = readers[-1].read_time_req()
            # new_label = dicts[-1].get(r[1], len(dicts[-1]))
            # dicts[-1][r[1]] = new_label
            # writers[-1].write((r[0], new_label))
            heapq.heappush(heap, (int(r[0]), r[1], len(readers)-1))

        while len(heap):
            # print("heap length {}".format(len(heap)))
            r = heapq.heappop(heap)
            idx = r[2]

            new_label = dicts[idx].get(r[1], )
            label_complete = dict_complete.get(r[1], len(dict_complete))
            dicts[idx][r[1]] = new_label
            dict_complete[r[1]] = label_complete

            writers[idx].write((int(r[0]), new_label))
            writer_complete.write((int(r[0]), label_complete))

            r = readers[idx].read_time_req()
            if r:
                heapq.heappush(heap, (int(r[0]), r[1], idx))

        for reader in readers:
            reader.close()
        for writer in writers:
            writer.close()
        writer_complete.close()
