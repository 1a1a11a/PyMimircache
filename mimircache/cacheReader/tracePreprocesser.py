# coding=utf-8
"""
a module for preprocessing tracefile
"""




import mmh3
from mimircache import *
from mimircache.cacheReader.binaryWriter import traceBinaryWriter



class tracePreprocessor:
    def __init__(self, reader):
        """

        :param reader:
        """
        self.reader = reader

        self.modulo = 2 ** 24
        self.N = len(reader)
        self.shards_file_loc = None



    def prepare_for_shards(self, sample_ratio=0.01, has_time=False, fmt=None, ofilename=None):
        written_records = 0     # number of records that are sampled out and written to file

        # it's better to detect and determine fmt here, which is not implemented yet
        if fmt is None:
            if has_time:
                fmt = "<LL"
            else:
                fmt = "<L"
        else:
            if has_time:
                assert len(fmt.strip("<#@!"))==2, "length of fmt is wrong, has_time enabled should be 2"
            else:
                assert len(fmt.strip("<#@!"))==1, "length of fmt is wrong, has_time disabled, should be 1"

        if ofilename is None:
            ofilename = "{}.shard.{}".format(os.path.basename(self.reader.file_loc), sample_ratio)
        self.shards_file_loc = ofilename

        mapping_dict = {}
        writer = traceBinaryWriter(ofilename, fmt=fmt)
        counter = 0     # number of unique labels
        if has_time:
            r = self.reader.read_time_request()
            init_time = r[0]
            while r:
                h = mmh3.hash(str(r[1])) % self.modulo
                if h <= self.modulo*sample_ratio:
                    if r[1] not in mapping_dict:
                        mapping_dict[r[1]] = counter
                        counter += 1
                    writer.write( (int(r[0] - init_time), mapping_dict[r[1]]) )
                    written_records += 1
                r = self.reader.read_time_request()
        else:
            r = self.reader.read_one_element()
            while r:
                h = mmh3.hash(str(r)) % self.modulo
                if h <= self.modulo*sample_ratio:
                    if r not in mapping_dict:
                        mapping_dict[r] = counter
                        counter += 1
                    writer.write( (mapping_dict[r], ) )
                    written_records += 1
                r = self.reader.read_one_element()

        writer.close()
        return (int(self.N * sample_ratio), written_records, ofilename, fmt)


    def close(self):
        os.remove(self.shards_file_loc)


if __name__ == "__main__":
    from mimircache.cacheReader.vscsiReader import vscsiReader
    from mimircache.cacheReader.csvReader import csvReader
    reader = vscsiReader("../data/trace.vscsi")
    reader2 = csvReader("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean",
                        init_params={"header":False, "delimiter":"\t", "label_column":5, 'real_time_column':1})
    print(tracePreprocessor(reader).prepare_for_shards(has_time=True, fmt="<LL"))






