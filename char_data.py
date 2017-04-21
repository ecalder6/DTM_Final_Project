# python3 data.py data/twitter.txt twitter.tfrecords

vocab = "| abcdefghijklmnopqrstuvwxyz"+\
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"+\
        "1234567890"+\
        "~`!@#$%^&*()_+-=[]{}:;\"'<>,./?\\"
tweet_len = 140

import random
import sys

import itertools
from collections import defaultdict
import tensorflow as tf

import numpy as np

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
    idx_q = np.zeros([data_len, tweet_len], dtype=np.int32) 
    idx_a = np.zeros([data_len, tweet_len], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, tweet_len)
        a_indices = pad_seq(atokenized[i], w2idx, tweet_len)
        # print(len(q_indices))
        # print(len(a_indices))

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
        indices.append(lookup[word])
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
    with open(input_filename, encoding="utf8") as f:
        i = 0
        skip = False
        qtokenized = []
        atokenized = []
        print('\n >> Reading and tokenizing words')
        for line in f.readlines():
            if skip:
                skip = False
                i += 1
                continue
            line = line.strip()

            tokens = []
            for char in line:
                if char in vocab:
                    tokens.append(char)
            if len(tokens) > tweet_len:
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

        print('\n >> Zero Padding')
        char_to_ix = { ch:i for i,ch in enumerate(vocab) }
        idx_q, idx_a = zero_pad(qtokenized, atokenized, char_to_ix)
        print(idx_q[:5])
        print(idx_a[:5])

        print('\n >> Convert to tfrecords and write to disk')        
        write_to_tfrecords(output_filename, idx_q, idx_a)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Specifiy Input_file_name and output_file_name")
        exit(0)
    process_data(sys.argv[1], sys.argv[2])