import json
import tensorflow as tf
import os

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  filename = name + '.tfrecords'
  writer = tf.python_io.TFRecordWriter(filename)

  for p in data_set:
    comment = p['body']
    comment = str.encode(comment)
    example = tf.train.Example(features=tf.train.Features(feature={
        'comment': _bytes_feature(comment)}))
    writer.write(example.SerializeToString())
  writer.close()

with open('example.json') as data_file:
    data = json.load(data_file)

    convert_to(data, "example")