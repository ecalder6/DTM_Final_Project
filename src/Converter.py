'''
Convert raw data to tfrecords
Currently only supports twitter data with format:
    Tweet
    Reply
'''

import nltk, pickle, itertools, string
import numpy as np
import tensorflow as tf

class Converter(object):
    def __init__(self, input_filename='', output_filename='', meta_file='', max_length=20, min_length=1, vocab_size=30000):
        self.en_whitelist = 'abcdefghijklmnopqrstuvwxyz '
        self.unk_token = 'unk'
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.max_length = max_length
        self.min_length = min_length
        self.vocab_size = vocab_size
        self.meta_file = meta_file

    def process_data(self):
        '''
        Takes in a text file and converts it into a tfrecord
        '''
        maxq = maxa = self.max_length
        minq = mina = self.min_length
        tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        translate_table = dict((ord(char), None) for char in string.punctuation)
        print("Beging tokenization")
        with open(self.input_filename, encoding="utf8") as f:
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
            print("Finished tokenization")
            print("Indexing")
            idx2w, w2idx, vocab = self.index(qtokenized + atokenized)
            print("Padding")
            idx_q, idx_a = self.zero_pad(qtokenized, atokenized, w2idx, maxq, maxa)

            # count of unknowns

            unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
            # count of words

            word_count = (idx_q > 1).sum() + (idx_a > 1).sum()
            # % unknown

            print('% unknown : {}'.format(100 * (unk_count/word_count)))

            print('\n >> Convert to tfrecords and write to disk')        
            self.write_to_tfrecords(self.output_filename, idx_q, idx_a)

            # let us now save the necessary dictionaries
            metadata = {
                    'w2idx' : w2idx,
                    'idx2w' : idx2w,
                    'vocab' : vocab,
                        }

            # write to disk : data control dictionaries
            with open(self.meta_file, 'wb') as f:
                pickle.dump(metadata, f)

    def index(self, tokenized_sentences):
        '''
        read list of words, create index to word,
        word to index dictionaries
            return tuple( vocab->(word, count), idx2w, w2idx )
        '''

        # get frequency distribution
        freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))

        # get vocabulary of 'vocab_size' most used words
        vocab = freq_dist.most_common(self.vocab_size)

        # index2word
        index2word = ["_"] + [self.unk_token] + [ x[0] for x in vocab ]

        # word2index
        word2index = dict([(w,i) for i,w in enumerate(index2word)] )
        vocab = list(map(lambda x: x[0], ["_"] + [self.unk_token] + vocab))
        return index2word, word2index, vocab

    def zero_pad(self, qtokenized, atokenized, w2idx, maxq, maxa):
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
            q_indices = self.pad_seq(qtokenized[i], w2idx, maxq)
            a_indices = self.pad_seq(atokenized[i], w2idx, maxa)

            #print(len(idx_q[i]), len(q_indices))
            #print(len(idx_a[i]), len(a_indices))
            idx_q[i] = np.array(q_indices)
            idx_a[i] = np.array(a_indices)

        return idx_q, idx_a
    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def write_to_tfrecords(self, output_filename, idx_q, idx_a):
        """Converts a dataset to tfrecords."""
        writer = tf.python_io.TFRecordWriter(output_filename)

        for q, a in zip(idx_q, idx_a):
            if not len(q):
                print("NOT LEN Q!")
            if not len(a):
                print("NOT LEN A!")
            example = tf.train.Example(features=tf.train.Features(feature={
                'question': self.int64_feature(q),
                'answer': self.int64_feature(a)}))
            writer.write(example.SerializeToString())
        writer.close()

    def pad_seq(self, seq, lookup, maxlen):
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
                indices.append(lookup[self.unk_token])
        return indices + [0]*(maxlen - len(seq))

    
    