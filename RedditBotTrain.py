
# coding: utf-8

# # Reddit Neural Bot Trainer
# -----
# #### ToDo
# - Subredding embeddings

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random
import time

import numpy as np
import tensorflow as tf


# # Data

# In[2]:

MAX_TWEET_LENGTH = 20
BATCH_SIZE = 100

# x = tf.Variable([1.0, 2.0])

# init = tf.global_variables_initializer()

# sess = tf.Session()
# sess.run(init)
# v = sess.run(x)    
# print(v) # will show you your variable.


# In[3]:

import pickle
meta = pickle.load( open( "metadata", "rb" ) )

vocab = meta["vocab"]
word_to_ix = meta["w2idx"]
ix_to_word = meta["idx2w"]

def to_eng(ids):
    return ' '.join([ix_to_word[id] if id != 0 else '' for id in ids])


# In[4]:

proto_files = glob.glob('./*.tfrecords')
random.shuffle(proto_files)


# In[5]:

filename_queue = tf.train.string_input_producer(proto_files)


# In[6]:

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
      #'subredddit_id': tf.FixedLenFeature([], tf.int64),
      'question': tf.FixedLenFeature([MAX_TWEET_LENGTH], tf.int64),
      'answer': tf.FixedLenFeature([MAX_TWEET_LENGTH], tf.int64),
  })


# normal_rv = tf.Variable( tf.truncated_normal([2,3],stddev = 0.1))

# #initialize the variable
# init_op = tf.global_variables_initializer()
# print(normal_rv)
# print(features["comment"])

# #run the graph
# with tf.Session() as sess:
#     sess.run(init_op) #execute init_op
#     #print the random values that we sample
#     print (sess.run(normal_rv))
#     print (sess.run(features["comment"]))


# In[7]:

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * BATCH_SIZE

tweet, replies = tf.train.shuffle_batch(
    [features['question'], features['answer']],
    batch_size=BATCH_SIZE, capacity=capacity, min_after_dequeue=min_after_dequeue)


# # Model

# In[8]:

LEARNING_RATE = 0.01
SEQ_MAX_LEN = MAX_TWEET_LENGTH
RNN_HIDDEN_SIZE = 1024
LAYERS = 3
EMB_SIZE = 256
VOCAB_SIZE = len(vocab)


# ### Encoding

# In[9]:

inner_cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN_SIZE)
enc_cell = tf.contrib.rnn.MultiRNNCell([inner_cell] * LAYERS)


# In[10]:

word_embeddings = tf.get_variable("embedding", [VOCAB_SIZE, EMB_SIZE])
emb_tweet = tf.nn.embedding_lookup(word_embeddings, tweet)


# In[11]:

_, thought_vector = tf.nn.dynamic_rnn(
    enc_cell, emb_tweet, swap_memory=True, dtype=tf.float32)


# ### Decoding

# In[12]:

reply_input = tf.concat(  # Add GO token to start
    [tf.zeros(shape=(BATCH_SIZE, 1), dtype=tf.int64), replies[:, :SEQ_MAX_LEN-1]], axis=1)
emb_reply_input = tf.nn.embedding_lookup(word_embeddings, reply_input)


# In[13]:

dec_cell = tf.contrib.rnn.OutputProjectionWrapper(enc_cell, VOCAB_SIZE)


# In[14]:

with tf.variable_scope("decoder"):
    dec_out, _ = tf.nn.dynamic_rnn(
        dec_cell, emb_reply_input, initial_state=thought_vector, swap_memory=True, dtype=tf.float32)


# In[15]:

xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_out, labels=replies)


# In[16]:

loss = tf.reduce_sum(xent, axis=[1])
ave_loss = tf.reduce_mean(loss)


# ### Sampling

# In[17]:

SAMPLE_TEMP = 0.7

def loop_fn(time, cell_output, cell_state, loop_state):
    if cell_output is None:  # time == 0
        next_cell_state = thought_vector  # state from the encoder
        next_input = tf.zeros([BATCH_SIZE], dtype=tf.int64)  # GO symbol
        next_input = tf.nn.embedding_lookup(word_embeddings, next_input)
        emit_output = tf.zeros([], dtype=tf.int64)
    else:
        next_cell_state = cell_state
        sample = tf.squeeze(tf.multinomial(cell_output / SAMPLE_TEMP, 1))
        print(sample)
        emb_sample = tf.nn.embedding_lookup(word_embeddings, sample)
        next_input = emb_sample
        emit_output = sample
    elements_finished = time >= tf.constant(SEQ_MAX_LEN, shape=(BATCH_SIZE,))
    finished = tf.reduce_all(elements_finished)
    print(next_input)
    next_input = tf.cond(
        finished,
        lambda: tf.zeros([BATCH_SIZE, EMB_SIZE], dtype=tf.float32),
        lambda: next_input)
    print(next_input)
    next_loop_state = None
    return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

with tf.variable_scope("decoder", reuse=True):
    outputs_ta, _, _ = tf.nn.raw_rnn(dec_cell, loop_fn, swap_memory=True)
    sample = outputs_ta.stack()


# # Training

# In[18]:

lr = tf.placeholder_with_default(LEARNING_RATE, [], name="lr")
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(ave_loss, tvars), 1.0)
optimizer = tf.train.RMSPropOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars))


# In[19]:

from functools import reduce
import operator

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


# In[20]:

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#coord.join(threads)
#sess.close()


# In[21]:

# CHECKPOINT_PATH = './checkpoints/'

# saver = tf.train.Saver()
# latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_PATH)
# if latest_checkpoint:
#     saver.restore(sess, latest_checkpoint)


# In[22]:

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
        c, r = sess.run([tweet, sample])
        for i in range(20):
            print("====================================")
            print(to_eng(c[i]), "-->", to_eng(r[:, i]))
            print("====================================")

        l_ave = b_ave = d_ave = 0
print("DONE")
#         saver.save(sess, CHECKPOINT_PATH + "checkpoint", global_step=0)
        #print('-'*24 + '|' + '-'*24 + '|' + '-'*24 + '|' + '-'*24 + '|')


# In[ ]:



