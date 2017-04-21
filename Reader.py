import os, random, time, glob, pickle, itertools, string

import nltk
from collections import defaultdict
import tensorflow as tf
import numpy as np


class Reader:

    def __init__(self, data_dir='./', max_length=20):
        en_whitelist = 'abcdefghijklmnopqrstuvwxyz '
        unk_token = 'unk'


        self._data_dir = data_dir
        self._max_length = max_length

    def process_data(self, input_filename, output_filename, vocab_size=20000,  \
                        meta_file='metadata', maxq=20, minq=1, maxa=20, mina=1):
        self._vocab_size = vocab_size
        self._max_length = maxq
        tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        translate_table = dict((ord(char), None) for char in string.punctuation)
        with open(input_filename, encoding="utf8") as f:
            i = 0
            # Skip 2 lines at a time
            skip = False
            qtokenized = []
            atokenized = []
            for line in f.readlines():
                if skip:
                    skip = False
                    i += 1
                    continue
                line = line.strip().lower()
                line = line.translate(translate_table)
                tokens = tknzr.tokenize(line)
                if len(tokens) > maxq:
                    skip = True
                    i += 1
                    continue
                if i % 2:
                    qtokenized.append(tokens)
                else:
                    atokenized.append(tokens)
                i += 1
            idx2w, w2idx, vocab = self._index( qtokenized + atokenized)

            idx_q, idx_a = self._zero_pad(qtokenized, atokenized, w2idx, maxq, maxa)

            # count of unknowns

            unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
            # count of words

            word_count = (idx_q > 1).sum() + (idx_a > 1).sum()
            # % unknown

            print('% unknown : {}'.format(100 * (unk_count/word_count)))

            print('\n >> Convert to tfrecords and write to disk')        
            self._write_to_tfrecords(output_filename, idx_q, idx_a)

            # let us now save the necessary dictionaries
            metadata = {
                    'w2idx' : w2idx,
                    'idx2w' : idx2w,
                    'vocab' : vocab,
                        }

            # write to disk : data control dictionaries
            with open(meta_file, 'wb') as f:
                pickle.dump(metadata, f)

    def _index(self, tokenized_sentences):
        '''
        read list of words, create index to word,
        word to index dictionaries
            return tuple( vocab->(word, count), idx2w, w2idx )
        '''

        # get frequency distribution
        freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))

        # get vocabulary of 'vocab_size' most used words
        vocab = freq_dist.most_common(self._vocab_size)

        # index2word
        index2word = ["_"] + [UNK] + [ x[0] for x in vocab ]

        # word2index
        word2index = dict([(w,i) for i,w in enumerate(index2word)] )
        vocab = list(map(lambda x: x[0], ["_"] + [UNK] + vocab))
        return index2word, word2index, vocab

    def _zero_pad(self, qtokenized, atokenized, w2idx, maxq, maxa):
        '''
        create the final dataset : 
        - convert list of items to arrays of indices
        - add zero padding
            return ( [array_en([indices]), array_ta([indices]) )
        
        '''
        # num of rows
        data_len = len(qtokenized)

        # numpy arrays to store indices
        idx_q = np.zeros([data_len, maxq], dtype=np.int32) 
        idx_a = np.zeros([data_len, maxa], dtype=np.int32)

        for i in range(data_len):
            q_indices = pad_seq(qtokenized[i], w2idx, maxq)
            a_indices = pad_seq(atokenized[i], w2idx, maxa)

            #print(len(idx_q[i]), len(q_indices))
            #print(len(idx_a[i]), len(a_indices))
            idx_q[i] = np.array(q_indices)
            idx_a[i] = np.array(a_indices)

        return idx_q, idx_a
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _write_to_tfrecords(self, output_filename, idx_q, idx_a):
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

    def _pad_seq(self, seq, lookup, maxlen):
        '''
        replace words with indices in a sequence
        replace with unknown if word not in lookup
            return [list of indices]

        '''
        indices = []
        for word in seq:
            if word in lookup:
                indices.append(lookup[word])
            else:
                indices.append(lookup[UNK])
        return indices + [0]*(maxlen - len(seq))

    def read_records(self, files=None, max_length=20):
        try:
            self._max_length = max_length
            proto_files = []
            if files:
                for f in files:
                    proto_files += glob.glob(self._data_dir+f)
            else:
                proto_files = glob.glob(self._data_dir + '*.tfrecords')
            # Construct a queue of records to read
            filename_queue = tf.train.string_input_producer(proto_files)

            # reader outputs the records from a TFRecords file
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            self.features = tf.parse_single_example(
                serialized_example,
                # Defaults are not specified since both keys are required.
                features={
                    #'subredddit_id': tf.FixedLenFeature([], tf.int64),
                    'question': tf.FixedLenFeature([max_length], tf.int64),
                    'answer': tf.FixedLenFeature([max_length], tf.int64),
                })
        except Exception as e:
            print("ERROR: Could not read records")

    def read_metadata(self, meta_file='metadata'):
        try:
            self._meta = pickle.load( open( DATA_DIR + "metadata", "rb" ) )
        except Exception as e:
            print("ERROR: Metadata file not found")

    def get_batches(self, features, batch_size=100, min_after_dequeue=10000):
        self._min_after_dequeue = min_after_dequeue
        self._capacity = min_after_dequeue + 3 * BATCH_SIZE
        self._batch_size = batch_size

        self._comment, self._replies = tf.train.shuffle_batch(
            [features['question'], features['answer']],
            batch_size=batch_size, capacity=self._capacity, min_after_dequeue=min_after_dequeue)