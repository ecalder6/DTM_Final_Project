from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Reader import Reader
from Converter import Converter
from LSTMVAE import LSTMVAE

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

    if args.run_converter:
        converter = Converter(input_filename = args.data_dir + 'twitter.txt', output_filename = args.data_dir + 'twitter.tfrecords', meta_file = args.data_dir + 'metadata')
        converter.process_data()

    # Set up and run reader
    reader = Reader(data_dir=args.data_dir)
    reader.read_metadata()
    tweets, replies = reader.read_records()

    seq_max_len = reader.max_length
    learning_rate = args.learning_rate


    # Set up model
    model = LSTMVAE(tweets, replies,
                reader.batch_size, args.emb_size,
                args.latent_size, reader.vocab_size, reader.max_length)
    ave_loss = model.loss
    lr = tf.placeholder_with_default(learning_rate, [], name="lr")
    train_op = model.train(lr)
    sample = model.sample(args.sample_temp)


    #print(print_shapes())

    # Set up training session
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Set up checkpoint
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)
    if latest_checkpoint:
        saver.restore(sess, latest_checkpoint)

    l_ave = b_ave = d_ave = 0

    UPDATE_EVERY = 100

    for step in range(args.iterations):
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
                # Windows: chcp 65001
                print(to_eng(c[i], reader.meta['idx2w']), "-->", to_eng(r[:, i], reader.meta['idx2w']))
                print("====================================")

            l_ave = b_ave = d_ave = 0
            saver.save(sess, args.checkpoint_path + "checkpoint", global_step=0)
    print("DONE")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=int)
    parser.add_argument('--rnn_hidden_size', default=512, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--emb_size', default=512, type=int)
    parser.add_argument('--keep_prob', default=0.8, type=float)
    parser.add_argument('--sample_temp', default=0.7, type=float)
    parser.add_argument('--latent_size', default=128, type=int)
    parser.add_argument('--iterations', default=5000, type=int)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--checkpoint_path', default='./twitter_checkpoints/', type=str)
    parser.add_argument('--run_converter', default=False, type=bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
