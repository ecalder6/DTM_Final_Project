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

    def __init__(self, data_dir='', max_length=20, min_length=1):
        self.data_dir = data_dir
        self.max_length = max_length
        self.min_length = min_length

    def read_records(self, files=None, batch_size=100, min_after_dequeue=10000):
        # try:
        proto_files = []
        if files:
            for f in files:
                proto_files += glob.glob(self._ata_dir+f)
        else:
            proto_files = glob.glob(self.data_dir + '*.tfrecords')
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
        tweets, replies = tf.train.shuffle_batch(
            [features['question'], features['answer']],
            batch_size=self.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        return tweets, replies
        # except Exception as e:
        #     print("ERROR: Could not read records")

    def read_metadata(self, meta_file='metadata'):
        try:
            self.meta = pickle.load( open( self.data_dir + "metadata", "rb" ) )
            self.vocab_size = len(self.meta['vocab'])
        except Exception as e:
            print("ERROR: Metadata file not found")