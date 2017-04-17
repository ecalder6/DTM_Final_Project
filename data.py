# python3 data.py data/twitter.txt twitter.tfrecords

EN_WHITELIST = 'abcdefghijklmnopqrstuvwxyz '
limit = {
        'maxq' : 20,
        'minq' : 1,
        'maxa' : 20,
        'mina' : 1
        }

UNK = 'unk'
VOCAB_SIZE = 20000

import random
import sys

import nltk
import itertools
from collections import defaultdict
import tensorflow as tf
import string

import numpy as np

import pickle
from nltk.tokenize import TweetTokenizer


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ["_"] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    vocab = list(map(lambda x: x[0], ["_"] + [UNK] + vocab))
    return index2word, word2index, vocab

'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a

'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_to_tfrecords(output_filename, idx_q, idx_a):
    """Converts a dataset to tfrecords."""
    writer = tf.python_io.TFRecordWriter(output_filename)

    for q, a in zip(idx_q, idx_a):
        if not len(q):
            print("NOT LEN Q!")
        if not len(a):
            print("NOT LEN A!")
        example = tf.train.Example(features=tf.train.Features(feature={
            'question': _int64_feature(q),
            'answer': _int64_feature(a)}))
        writer.write(example.SerializeToString())
    writer.close()

def process_data(input_filename, output_filename):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    with open(input_filename, encoding="utf8") as f:
        i = 0
        # Skip 2 lines at a time
        skip = False
        qtokenized = []
        atokenized = []
        print('\n >> Reading and tokenizing words')
        for line in f.readlines():
            if skip:
                skip = False
                i += 1
                continue
            line = line.strip().lower()
            line = line.translate(translate_table)
            tokens = tknzr.tokenize(line)
            if len(tokens) > limit['maxq']:
                skip = True
                i += 1
                continue
            if i % 2:
                qtokenized.append(tokens)
            else:
                atokenized.append(tokens)
            i += 1
        print(qtokenized[:5])
        print(atokenized[:5])
        print('\n >> Index words')
        idx2w, w2idx, vocab = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)
        print(vocab[:5])

        print('\n >> Zero Padding')
        idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)
        print(idx_q[:5])
        print(idx_a[:5])

        # count of unknowns

        unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
        # count of words

        word_count = (idx_q > 1).sum() + (idx_a > 1).sum()
        # % unknown

        print('% unknown : {}'.format(100 * (unk_count/word_count)))


        print('\n >> Convert to tfrecords and write to disk')        
        write_to_tfrecords(output_filename, idx_q, idx_a)

        # let us now save the necessary dictionaries
        metadata = {
                'w2idx' : w2idx,
                'idx2w' : idx2w,
                'vocab' : vocab,
                    }

        # write to disk : data control dictionaries
        with open('metadata', 'wb') as f:
            pickle.dump(metadata, f)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Specifiy Input_file_name and output_file_name")
        exit(0)
    process_data(sys.argv[1], sys.argv[2])