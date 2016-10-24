

import os
from mimircache import vscsiReader



def vReaderToPReader(file_path, output_path):
    vReader = vscsiReader(file_path)
    with open(output_path, 'w') as ofile:
        for req in vReader:
            ofile.write("{}\n".format(req))

def vReaderTocsvReader(file_pth, output_path):
    pass


def splitTrace(reader, n, output_folder, prefix=""):
    """
    given a reader, split the content into n small files
    :param reader:
    :param n:
    :param output_folder:
    :param prefix:
    :return:
    """
    total_num = reader.get_num_of_total_requests()
    num_each_file = total_num // n
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ofiles = []
    for i in range(n):
        ofiles.append(open("{}/{}".format(output_folder, prefix+str(i)), 'w'))
    file_p = -1
    for i, e in enumerate(reader):
        if i % num_each_file == 0 and file_p!=len(ofiles)-1:
            file_p += 1
        ofiles[file_p].write("{}\n".format(e))

    for i in range(n):
        ofiles[i].close()





if __name__ == "__main__":
    import sys
    # vReaderToPReader(sys.argv[1], sys.argv[2])
    reader = vscsiReader("../data/traces/w38_vscsi1.vscsitrace")
    splitTrace(reader, 2, "w38Split")
