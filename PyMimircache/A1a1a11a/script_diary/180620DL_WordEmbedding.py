

import os, sys, time
from collections import deque, defaultdict, Counter
import numpy as np
from PyMimircache.bin.conf import *
from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
import bisect
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pprint import pprint

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

from tqdm import tqdm
sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
from PyMimircache.A1a1a11a.myUtils.DLUtils import get_txt_trace, read_data, gen_data_binary, gen_data_rand_binary


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def word_embedding(dat, dat_type, set_size=2000, window_size=12, embedding_size=300):
    if not os.path.exists(dat):
        get_txt_trace(dat, dat_type)
    with open(dat) as ifile:
        all_data = ifile.read().splitlines()
    # print(len(all_data))
    # tk = Tokenizer(num_words=2000)
    # tk.fit_on_texts(tqdm(all_data, desc="Tokenizing"))
    # sequences = np.array(tk.texts_to_sequences(all_data))
    # print(sequences)
    # vocabulary_size = len(tk.word_index) + 1

    data, count, dictionary, reversed_dictionary = build_dataset(all_data, 2000)
    print(len(data))

    sampling_table = sequence.make_sampling_table(set_size)
    couples, labels = skipgrams(data, set_size, window_size=window_size, sampling_table=sampling_table)

    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")
    print(couples[:10], labels[:10])


    input_target = Input((1,))
    input_context = Input((1,))
    embedding = Embedding(set_size, embedding_size, input_length=1, name='embedding')
    target = embedding(input_target)
    target = Reshape((embedding_size, 1))(target)
    context = embedding(input_context)
    context = Reshape((embedding_size, 1))(context)

    # setup a cosine similarity operation which will be output in a secondary model
    similarity = merge([target, context], mode='cos', dot_axes=0)

    # now perform the dot product operation to get a similarity measure
    dot_product = merge([target, context], mode='dot', dot_axes=1)
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    # create the primary training model
    model = Model(input=[input_target, input_context], output=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # create a secondary validation model to run our similarity checks during training
    validation_model = Model(input=[input_target, input_context], output=similarity)


def myt():
    from numpy import array
    # define documents
    docs = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor work',
            'Could have done better.']
    # define class labels
    labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # integer encode the documents
    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    # print(encoded_docs)
    # pad documents to a max length of 4 words
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)
    # define the model
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

def myt2(dat):
    get_txt_trace(dat, "cphy")
    d = defaultdict(int)
    with open(dat) as ifile:
        for line in ifile:
            d[line.strip()] += 1
    l = sorted(d.items(), key=lambda i: i[1], reverse=True)
    print("20 {} 200 {} 2000 {} 20000 {}".format(l[20], l[200], l[2000], l[20000]))

