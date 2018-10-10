# coding=utf-8
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.pool import Pool

TRAIN_RATIO = 0.3
CUTOFF = 7
FEATURE_FOLDER = "features/specific_rd/"
OUTPUT_FOLDER = "SLRU_DATA/rd/"


import os, sys, math, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from PyMimircache import *
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
from PyMimircache.bin.conf import initConf

################################## Global variable and initialization ###################################

TRACE_TYPE = "cloudphysics"

TRACE_DIR, NUM_OF_THREADS = initConf(TRACE_TYPE, trace_format='variable')



################################## SLRU  ###################################

class SLRU_ML:
    def __init__(self, dat, vscsi_type=-1):
        assert "vscsi" in dat, "vscsi type {}, but data({}) doesn't qualify".format(vscsi_type, dat)
        if vscsi_type == -1:
            vscsi_type = int(dat[dat.find('vscsi')+5])

        if vscsi_type == 1:
            self.fmt = "<3I2H2Q"
        elif vscsi_type == 2:
            self.fmt = "<2H3I3Q"
        else:
            print("unknown vscsi type {}".format(vscsi_type))
            return

        self.dat = dat
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)
        self.trace_num = os.path.basename(self.dat)
        self.trace_num = self.trace_num[:self.trace_num.find('_')]
        self.reader = BinaryReader(self.dat, open_c_reader=True,
                                   init_params={"label":6, "real_time":7, "fmt": self.fmt})
        self.writer = TraceBinaryWriter("{}/{}.vscsi".format(OUTPUT_FOLDER, self.trace_num), self.fmt)
        # print(dat)


    def output_trace(self, range):
        assert len(range) == 2, "range must be in the form of (min, max)"
        self.reader.reset()
        n = 0
        maxLine = range[1]
        if maxLine == -1:
            maxLine = len(self.reader)

        # print("output range {}".format(range))
        line_count = 0
        line = self.reader.read_complete_req()
        while line:
            line_count += 1
            if line_count - 1 < range[0]:  # -1 because we add 1 before checking
                line = self.reader.read_complete_req()
                continue
            if line_count - 1 >= maxLine:
                break
            self.writer.write(line)
            n += 1
            line = self.reader.read_complete_req()
        # print("output {} lines".format(n))
        self.reader.reset()


    def prepare_specific(self):
        my_data = np.genfromtxt("{}/{}_t2000.csv".format(FEATURE_FOLDER, self.trace_num),
                                delimiter=',')
        assert len(my_data)==len(self.reader), "number of requests in csv is not " \
        "equal to len(reader), {}:{}".format(len(my_data), len(self.reader))
        cutoff = len(self.reader) // CUTOFF

        y = my_data[cutoff:-cutoff, 0].astype(int)
        x = my_data[cutoff:-cutoff, 1:]

        # transform y into category by doing log4, then -5

        self.maxY = np.max(y)
        y[y==-1] = self.maxY * 4
        y = (np.log2(y + 2)/2 - 5).astype(int)
        y[y<0] = 0

        self.maxX = np.max(x[:, 0])
        x[:, 0][x[:, 0]==-1] =  self.maxX * 4
        x[:, 0] = (np.log2(x[:, 0] + 2)/2 - 5).astype(int)
        x[:, 0][x[:, 0]<0] = 0

        # treat -1 as maxY+1
        # maxY = max(y)
        # y[y==-0] = maxY + 1


        n = len(y)
        train_size = int(n * TRAIN_RATIO)
        self.output_trace((cutoff+train_size, len(self.reader)-cutoff))

        train_x = x[: train_size]
        train_y = y[: train_size]
        filter_out = np.logical_not(np.logical_or(train_x[:, 0] == -1, train_y[:] == -1))
        train_x = train_x[filter_out]
        train_y = train_y[filter_out]

        val_x = x[train_size:]
        val_y = y[train_size:]

        # print("cutoff "+str(cutoff))
        # print("y: {}".format(len(y)))
        # print("train y: {}".format(len(train_y)))
        # print("val y: {}".format(len(val_y)))
        # print("my data: {}".format(len(my_data)))
        # print("reader: {}".format(len(self.reader)))

        # reg = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression(n_jobs=-1))
        # reg = LogisticRegression(n_jobs=-1)
        reg = RandomForestClassifier(n_jobs=4)

        reg.fit(train_x, train_y)
        pred_y = reg.predict(val_x)

        # calculate +-1 accuracy
        correct = 0
        for y_p, y_t in zip(pred_y, val_y):
            if abs(y_p - y_t) <= 1:
                correct += 1

        accuracy = correct / len(pred_y)

        print("{}(specific): {}, {}, +-1 accuracy {}".format(
            self.trace_num, reg.score(train_x, train_y),
            reg.score(val_x, val_y), accuracy))


        with open("{}/{}.y".format(OUTPUT_FOLDER, self.trace_num), 'wb') as ofile:
            for y in pred_y:
                if y >= 0:
                    ofile.write(struct.pack("<b", (int)(y)))
                else:
                    ofile.write(struct.pack("<b", (int)(0)))

        if self.writer:
            self.writer.close()
            self.write = None
        return (cutoff+train_size, len(self.reader)-cutoff)


    def verify(self):
        with BinaryReader("{}/{}.vscsi".format(OUTPUT_FOLDER, self.trace_num),
                          init_params={"label":6, "real_time":7, "fmt": self.fmt}) as reader:
            assert len(reader) == os.path.getsize("{}/{}.y".format(OUTPUT_FOLDER, self.trace_num)), \
                "verification failed, {}:{}".format(len(reader),
                                                    os.path.getsize("{}/{}.y".format(OUTPUT_FOLDER, self.trace_num)))

    def write_real_hint(self, range):
        # range = (40668, 97605)
        line_count = 0
        with open("{}/{}_t2000.csv".format(FEATURE_FOLDER, self.trace_num)) as ifile:
            with open("{}/{}.realY".format(OUTPUT_FOLDER, self.trace_num), 'wb') as ofile:
                for y in ifile:
                # ofile.write("{}\n".format(y))
                    line_count += 1
                    if line_count - 1 < range[0]:
                        continue
                    if line_count - 1 >= range[1]:
                        continue

                    y = int(y.split(',')[0])



            ######################## for forward reuse distance ######################
                    if y < 0:
                        y = self.maxY * 4
                    y = int(math.log2(y+2) / 2 - 5)
                    if y < 0:
                        y = 0

            ######################## for forward distance ######################
                    # if y < 0:
                    #     y = -1
                    # else:
                    #     y = int(math.log2(y+2) / 2 - 5)
                    #     if y < 0:
                    #         y = 0
                # print(y)
                    ofile.write(struct.pack("<b", y))



    def prepare_interval(self):
        pass


    def __del__(self):
        try:
            if self.writer:
                self.writer.close()
                self.writer = None
        except Exception as e:
            print("Error when closing writer {}".format(e))




class accuracyCal:
    def __init__(self, dat):
        self.dat = dat
        self.predY = "{}/{}.y".format("SLRU_DATA", dat)
        self.realY = "{}/{}.realY".format("SLRU_DATA", dat)
        self.fmt = "<b"
        self.structIns = struct.Struct(self.fmt)
        self.lPred = []
        self.lReal = []
        with open(self.predY, 'rb') as pfile:
            b = pfile.read(1)
            while len(b):
                self.lPred.append(self.structIns.unpack(b)[0])
                b = pfile.read(1)

        with open(self.realY, 'rb') as rfile:
            b = rfile.read(1)
            while len(b):
                self.lReal.append(self.structIns.unpack(b)[0])
                b = rfile.read(1)



    def cal(self):
        accurate  = 0
        accurate1 = 0
        maxReal = max(self.lReal)
        for x, y in zip(self.lPred, self.lReal):
            if x == y:
                accurate += 1
            elif y==-1 and x>maxReal:
                accurate += 1
            if abs(x-y) <= 1:
                accurate1 += 1
            elif y==-1 and x>=maxReal:
                accurate1 += 1

        print("{}: accuracy {:.4f}, +-1 accuracy {:.4f}".format(self.dat,
                                                                accurate/len(self.lPred),
                                                                accurate1/len(self.lPred)))


    def plot(self, figname=None):
        if figname == None:
            figname = "{}.pred.real.png".format(self.dat)
        lReal = [x+random.randint(-10000, 10000)/20000 for n, x in enumerate(self.lReal) if n%10==0]
        lPred = [x+random.randint(-10000, 10000)/20000 for n, x in enumerate(self.lPred) if n%10==0]

        plt.scatter(lReal, lPred, s=0.2)
        plt.xlabel("real")
        plt.ylabel("pred")
        plt.savefig(figname, dpi=600)
        plt.clf()



def batch(dat):
    trace_num = os.path.basename(dat)
    trace_num = trace_num[:trace_num.find('_')]
    if os.path.exists("{}/{}.realY".format(OUTPUT_FOLDER, trace_num)):
        st = os.stat("{}/{}.realY".format(OUTPUT_FOLDER, trace_num))
        if st.st_size != 0:
            return

    p = SLRU_ML("{}/{}".format(TRACE_DIR, dat))
    range = p.prepare_specific()
    p.verify()
    p.write_real_hint(range)
    return


def accuracy(dat):
    ac = accuracyCal(dat)
    ac.cal()
    ac.plot()

def batch_accuracy():
    for i in range(1, 107):
        if i < 10:
            i = '0' + str(i)
        if os.path.exists("w{}.pred.real.png".format(i)):
            continue
        if os.path.exists("{}/w{}.realY".format("SLRU_DATA", i)):
            st = os.stat("{}/w{}.realY".format("SLRU_DATA", i))
            if st.st_size != 0:
                accuracy("w{}".format(i))


if __name__ == "__main__":

    p = SLRU_ML("{}/{}".format(TRACE_DIR, "w63_vscsi1.vscsitrace"))
    # p = SLRU_ML("{}/{}".format(TRACE_DIR, "test_vscsi1.vscsitrace"))
    range = p.prepare_specific()
    p.verify()
    p.write_real_hint(range)

    # accuracy("w106")
    # batch_accuracy()

    sys.exit(-1)

    # with Pool(4) as p:
    #     p.map(batch, [f for f in os.listdir(TRACE_DIR) if f.endswith("vscsitrace") and 'vscsi' in f])


    with ProcessPoolExecutor(8) as p:
        futures = {p.submit(batch, f) : f for f in os.listdir(TRACE_DIR) if f.endswith("vscsitrace") and 'vscsi' in f}
        for future in as_completed(futures):
            print(futures[future])

