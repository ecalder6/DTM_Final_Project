'''
Reader for our final project.
Deals with handling all the data reading
'''

import glob, pickle
import tensorflow as tf

class Reader(object):
    '''
    Reader class
    '''

    def __init__(self, data_dir='', max_length=20, min_length=1, task='twitter'):
        self.data_dir = data_dir
        self.max_length = max_length
        self.min_length = min_length
        self._task = task

    def read_records(self, files=None, train=True, batch_size=100, min_after_dequeue=10000):
        d = self.data_dir + 'train/' + self._task + '.tfrecords' if train else self.data_dir + 'test/' + self._task + '.tfrecords'
        proto_files = glob.glob(d)
        # Construct a queue of records to read
        filename_queue = tf.train.string_input_producer(proto_files)

        # reader outputs the records from a TFRecords file
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Select which attributes to use depending on which task
        attributes = None
        if self._task == 'twitter':
            attributes = { 'question': tf.FixedLenFeature([self.max_length], tf.int64), 'answer': tf.FixedLenFeature([self.max_length], tf.int64), }
        elif self._task == 'movie':
            attributes = { 'line': tf.FixedLenFeature([self.max_length], tf.int64)}

        features = tf.parse_single_example(
            serialized_example,
            features=attributes)
        capacity = min_after_dequeue + 3 * batch_size
        self.batch_size = batch_size
        if self._task == 'twitter':
            tweets, replies = tf.train.shuffle_batch(
                [features[k] for k in attributes.keys()],
                #[features['question'], features['answer']],
                batch_size=self.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
            return tweets, replies
        elif self._task == 'movie':
            lines = tf.train.shuffle_batch(
                [features[k] for k in attributes.keys()],
                batch_size=self.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
            return lines

    def read_metadata(self, meta_dir='metadata'):
        self.meta = pickle.load( open( self.data_dir + "metadata/" + self._task + "_" +  meta_dir, "rb" ) )
        self.vocab_size = len(self.meta['vocab'])