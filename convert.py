import json
import tensorflow as tf
import os

vocab = "| abcdefghijklmnopqrstuvwxyz"+\
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"+\
        "1234567890"+\
        "~`!@#$%^&*()_+-=[]{}:;\"'<>,./?\\"
char_to_ix = { ch:i for i,ch in enumerate(vocab) }

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  filename = name + '.tfrecords'
  writer = tf.python_io.TFRecordWriter(filename)

  for p in data_set:
    comment = p['body']
    in_vec = []
    i = 0
    for c in comment:
      if i == 50:
        break
      if c in char_to_ix:
        in_vec.append(char_to_ix[c])
      else:
        in_vec.append(1)
      i += 1
    in_vec += [1] * (50 - len(in_vec))
    print(len(in_vec))
    example = tf.train.Example(features=tf.train.Features(feature={
        'comment': _int64_feature(in_vec)}))
    writer.write(example.SerializeToString())
  writer.close()

with open('example.json') as data_file:
    data = json.load(data_file)

    convert_to(data, "example")
    print("Done")