from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random
import time

import numpy as np
import tensorflow as tf
import pickle
from functools import reduce
import operator

MAX_COMMENT_LENGTH = 20
BATCH_SIZE = 100


meta = pickle.load( open( "metadata", "rb" ) )

vocab = meta["vocab"]
word_to_ix = meta["w2idx"]
ix_to_word = meta["idx2w"]

def to_eng(ids):
    return ' '.join([ix_to_word[id] if id != 0 else '' for id in ids])

proto_files = glob.glob('twitter.tfrecords')
random.shuffle(proto_files)
filename_queue = tf.train.string_input_producer(proto_files)  #num_epochs=

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
      #'subredddit_id': tf.FixedLenFeature([], tf.int64),
      'question': tf.FixedLenFeature([MAX_COMMENT_LENGTH], tf.int64),
      'answer': tf.FixedLenFeature([MAX_COMMENT_LENGTH], tf.int64),
  })

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * BATCH_SIZE

comment, replies = tf.train.shuffle_batch(
    [features['question'], features['answer']],
    batch_size=BATCH_SIZE, capacity=capacity, min_after_dequeue=min_after_dequeue)

LEARNING_RATE = 0.01
SEQ_MAX_LEN = MAX_COMMENT_LENGTH
RNN_HIDDEN_SIZE = 1024
LAYERS = 3
CHAR_EMB_SIZE = 128
VOCAB_SIZE = len(vocab)

inner_cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN_SIZE)
enc_cell = tf.contrib.rnn.MultiRNNCell([inner_cell] * LAYERS)

char_embeddings = tf.get_variable("embedding", [VOCAB_SIZE, CHAR_EMB_SIZE])
emb_comment = tf.nn.embedding_lookup(char_embeddings, comment)

_, thought_vector = tf.nn.dynamic_rnn(
    enc_cell, emb_comment, swap_memory=True, dtype=tf.float32)

reply_input = tf.concat(  # Add GO token to start
    [tf.zeros(shape=(BATCH_SIZE, 1), dtype=tf.int64), replies[:, :SEQ_MAX_LEN-1]], axis=1)
emb_reply_input = tf.nn.embedding_lookup(char_embeddings, reply_input)

dec_cell = tf.contrib.rnn.OutputProjectionWrapper(enc_cell, VOCAB_SIZE)

with tf.variable_scope("decoder"):
    dec_out, _ = tf.nn.dynamic_rnn(
        dec_cell, emb_reply_input, initial_state=thought_vector, swap_memory=True, dtype=tf.float32)

xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_out, labels=replies)

loss = tf.reduce_sum(xent, axis=[1])
ave_loss = tf.reduce_mean(loss)



SAMPLE_TEMP = 0.7

def loop_fn(time, cell_output, cell_state, loop_state):
    if cell_output is None:  # time == 0
        next_cell_state = thought_vector  # state from the encoder
        next_input = tf.zeros([BATCH_SIZE], dtype=tf.int64)  # GO symbol
        next_input = tf.nn.embedding_lookup(char_embeddings, next_input)
        emit_output = tf.zeros([], dtype=tf.int64)
    else:
        next_cell_state = cell_state
        sample = tf.squeeze(tf.multinomial(cell_output / SAMPLE_TEMP, 1))
        print(sample)
        emb_sample = tf.nn.embedding_lookup(char_embeddings, sample)
        next_input = emb_sample
        emit_output = sample
    elements_finished = time >= tf.constant(SEQ_MAX_LEN, shape=(BATCH_SIZE,))
    finished = tf.reduce_all(elements_finished)
    print(next_input)
    next_input = tf.cond(
        finished,
        lambda: tf.zeros([BATCH_SIZE, CHAR_EMB_SIZE], dtype=tf.float32),
        lambda: next_input)
    print(next_input)
    next_loop_state = None
    return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

with tf.variable_scope("decoder", reuse=True):
    outputs_ta, _, _ = tf.nn.raw_rnn(dec_cell, loop_fn, swap_memory=True)
    sample = outputs_ta.stack()

lr = tf.placeholder_with_default(LEARNING_RATE, [], name="lr")
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(ave_loss, tvars), 1.0)
optimizer = tf.train.RMSPropOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars))




def print_shapes():
    train_vars = tf.trainable_variables()
    
    lines = ['']
    lines.append('Trainable Variables:')
    lines.append('====================')
    total_params = 0
    for var in train_vars:
        n_param = reduce(operator.mul, var.get_shape().as_list(), 1)
        total_params += n_param
        lines.append('%20s %8d %s' % (var.get_shape().as_list(), n_param, var.name))
    lines.append('Total trainable parameters: %d' % total_params)
    
    lines.append('')
    lines.append('Other Varaibles:')
    lines.append('================')
    total_params = 0
    for var in tf.global_variables():
        if var in train_vars: continue
        n_param = reduce(operator.mul, var.get_shape().as_list(), 1)
        total_params += n_param
        lines.append('%20s %8d %s' % (var.get_shape().as_list(), n_param, var.name))
    lines.append('Total non-trainable parameters: %d' % total_params)
    
    return '\n'.join(lines)

print(print_shapes())

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#coord.join(threads)
#sess.close()

l_ave = b_ave = d_ave = 0

UPDATE_EVERY = 100

for step in range(500):
    start_time = time.time()
    _, l = sess.run([train_op, ave_loss], {
        lr: 0.0001
    })
    duration = time.time() - start_time

    l_ave += l
    b_ave += l / SEQ_MAX_LEN / np.log(2.)
    d_ave += duration
    
    #print("|", end="")
    if step % UPDATE_EVERY == 0:
        print()
        l_ave = l_ave / UPDATE_EVERY if step else l_ave
        b_ave = b_ave / UPDATE_EVERY if step else b_ave
        d_ave = d_ave / UPDATE_EVERY if step else d_ave
        
        print(step)
        print(l_ave, "(", b_ave, ")\t|", "%.3f sec" % d_ave)
        c, r = sess.run([comment, sample])
        for i in range(20):
            print("====================================")
            print(to_eng(c[i]), "-->", to_eng(r[:, i]))
            print("====================================")

        l_ave = b_ave = d_ave = 0
print("DONE")
#         saver.save(sess, CHECKPOINT_PATH + "checkpoint", global_step=0)
        #print('-'*24 + '|' + '-'*24 + '|' + '-'*24 + '|' + '-'*24 + '|')