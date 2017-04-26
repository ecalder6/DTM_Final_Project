from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Reader import Reader
from Converter import Converter
from LSTMVAE import LSTMVAE

import os, random, time, glob, pickle

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



def main():
    args = get_args()

    if args.run_converter:
        converter = Converter(input_filename = args.data_dir + 'twitter.txt', output_filename = args.data_dir + 'twitter.tfrecords', meta_file = args.data_dir + 'metadata')
        converter.process_data()

    # Set up and run reader
    reader = Reader(data_dir=args.data_dir)
    reader.read_metadata()
    input, truth = reader.read_records()

    seq_max_len = reader.max_length
    learning_rate = args.learning_rate


    # Set up model
    model = LSTMVAE(tweets, \
                reader.batch_size, args.emb_size, \
                args.latent_size, reader.vocab_size, reader.max_length)
    # out = model.get_outputs(replies)
    # out = model.sample_test(args.sample_temp)
    # print(out)

    # loss = model.get_loss(replies, out, use_mutual=args.use_mutual)

    # Set up checkpoint
    # saver = tf.train.Saver()
    # latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)
    # if latest_checkpoint:
    #     saver.restore(sess, latest_checkpoint)

    ### Main loop for training
    UPDATE_EVERY = 10
    objective_loss = []
    kl_loss = []
    duration = time.time()
    c, r = sess.run([tweets, sample])
    for i in range(5):
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
    parser.add_argument('--use_mutual', default=False, type=bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
