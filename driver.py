from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Reader import Reader
from Seq2Seq import Seq2Seq

import os, random, time, glob, pickle

import numpy as np
import tensorflow as tf

from functools import reduce
import operator
import argparse


def to_eng(ids, ix_to_word):
    output = ""
    for id in ids:
        if id:
            output += ix_to_word[id] + " "
    return output


# def print_shapes():
#     train_vars = tf.trainable_variables()
#     lines = ['']
#     lines.append('Trainable Variables:')
#     lines.append('====================')
#     total_params = 0
#     for var in train_vars:
#         n_param = reduce(operator.mul, var.get_shape().as_list(), 1)
#         total_params += n_param
#         lines.append('%20s %8d %s' % (var.get_shape().as_list(), n_param, var.name))
#     lines.append('Total trainable parameters: %d' % total_params)
    
#     lines.append('')
#     lines.append('Other Varaibles:')
#     lines.append('================')
#     total_params = 0
#     for var in tf.global_variables():
#         if var in train_vars: continue
#         n_param = reduce(operator.mul, var.get_shape().as_list(), 1)
#         total_params += n_param
#         lines.append('%20s %8d %s' % (var.get_shape().as_list(), n_param, var.name))
#     lines.append('Total non-trainable parameters: %d' % total_params)
    
#     return '\n'.join(lines)




def main():
    args = get_args()
    reader = Reader(data_dir='./data/')
    reader.read_metadata()
    tweets, replies = reader.read_records()

    seq_max_len = reader.max_length
    learning_rate = args.learning_rate



    seq2seq = Seq2Seq(tweets, replies, args.rnn_hidden_size, args.layers, reader.batch_size, \
                args.emb_size, reader.vocab_size, seq_max_len)
    ave_loss = seq2seq.get_loss(replies)
    lr = tf.placeholder_with_default(learning_rate, [], name="lr")
    train_op = seq2seq.train(ave_loss, lr)
    sample = seq2seq.sample(args.sample_temp)


    #print(print_shapes())

    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    l_ave = b_ave = d_ave = 0

    UPDATE_EVERY = 100

    for step in range(500):
        start_time = time.time()
        _, l = sess.run([train_op, ave_loss], {
            lr: 0.0001
        })
        duration = time.time() - start_time

        l_ave += l
        b_ave += l / seq_max_len / np.log(2.)
        d_ave += duration
        
        #print("|", end="")
        if step % UPDATE_EVERY == 0:
            print()
            l_ave = l_ave / UPDATE_EVERY if step else l_ave
            b_ave = b_ave / UPDATE_EVERY if step else b_ave
            d_ave = d_ave / UPDATE_EVERY if step else d_ave
            
            print(step)
            print(l_ave, "(", b_ave, ")\t|", "%.3f sec" % d_ave)
            c, r = sess.run([tweets, sample])
            for i in range(20):
                print("====================================")
                print(to_eng(c[i], reader.meta['idx2w']), "-->", to_eng(r[:, i], reader.meta['idx2w']))
                print("====================================")

            l_ave = b_ave = d_ave = 0
    print("DONE")
    #         saver.

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=int)
    parser.add_argument('--rnn_hidden_size', default=512, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--emb_size', default=512, type=int)
    parser.add_argument('--keep_prob', default=0.8, type=float)
    parser.add_argument('--sample_temp', default=0.7, type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
