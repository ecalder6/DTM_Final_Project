'''
Reader for our final project.
Deals with handling all the data reading, writing, pickling.
'''

import glob
import pickle
import itertools
import string
import nltk
import tensorflow as tf
import numpy as np


class Reader(object):
    '''
    Reader class
    '''

    def __init__(self, data_dir='', max_length=20, min_length=1):
        en_whitelist = 'abcdefghijklmnopqrstuvwxyz '
        unk_token = 'unk'
        self._data_dir = data_dir
        self.max_length = max_length
        self.min_length = min_length

    def process_data(self, input_filename, output_filename, meta_file='metadata'):
        '''
        Takes in a text file and converts it into a tfrecord
        '''
        maxq = maxa = self._max_length
        minq = mina = self._min_length
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
            idx2w, w2idx, vocab = self._index(qtokenized + atokenized)

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

    
    
    def read_records(self, files=None, batch_size=100, min_after_dequeue=10000):
        # try:
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

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                #'subredddit_id': tf.FixedLenFeature([], tf.int64),
                'question': tf.FixedLenFeature([self.max_length], tf.int64),
                'answer': tf.FixedLenFeature([self.max_length], tf.int64),
            })
        capacity = min_after_dequeue + 3 * batch_size
        self.batch_size = batch_size
        comment, replies = tf.train.shuffle_batch(
            [features['question'], features['answer']],
            batch_size=self.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        return comment, replies
        # except Exception as e:
        #     print("ERROR: Could not read records")

    def read_metadata(self, meta_file='metadata'):
        try:
            self.meta = pickle.load( open( self._data_dir + "metadata", "rb" ) )
            self.vocab_size = len(self.meta['vocab'])
        except Exception as e:
            print("ERROR: Metadata file not found")