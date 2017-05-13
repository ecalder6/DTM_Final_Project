from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Reader import Reader
from RNNLM import RNNLMVAE

import os, random, time, glob, pickle

import tensorflow as tf

from functools import reduce
import operator
import argparse
import csv
import numpy as np

def to_eng(ids, ix_to_word):
    output = ""
    for id in ids:
        if id:
            output += ix_to_word[id] + " "
    return output

def main():
    args = get_args()

    # Set up and run reader
    reader = Reader(task=args.task, data_dir=args.data_dir, batch_size=args.batch_size, save_z=args.save_z)
    reader.read_metadata()
    lines, replies = None, None
    if args.task == "twitter":
        lines, replies = reader.read_records()
    elif args.task == "movie":
        lines = reader.read_records()
        replies = lines
    learning_rate = args.learning_rate

    # Set up model
    model = RNNLMVAE(args.batch_size, args.emb_size, args.latent_size, reader.vocab_size, reader.max_length)
    model.construct_graph(lines)
    model.train(lines,replies, args.iterations, reader.meta)

    print("DONE")

def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--mutual_loss_lambda', default=0.1, type=float)
    parser.add_argument('--rnn_hidden_size', default=512, type=int)
    parser.add_argument('--emb_size', default=512, type=int)
    parser.add_argument('--keep_prob', default=0.8, type=float)
    parser.add_argument('--sample_temp', default=0.7, type=float)
    parser.add_argument('--latent_size', default=512, type=int)
    parser.add_argument('--iterations', default=5000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--checkpoint_path', default='../twitter_checkpoint_vae_highway/', type=str)
    parser.add_argument('--use_mutual', default=False, type=str2bool)
    parser.add_argument('--use_vae', default=True, type=str2bool)
    parser.add_argument('--use_highway', default=True, type=str2bool)
    parser.add_argument('--task', default="twitter", type=str)
    parser.add_argument('--update_every', default=100, type=int)
    parser.add_argument('--use_checkpoint', default=True, type=str2bool)
    parser.add_argument('--save_z', default=False, type=str2bool)
    parser.add_argument('--kl_anneal', default=False, type=str2bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Sample command for movie:
    #   python train.py --checkpoint_path=../movie_lstm_checkpoint/ --use_mutual=False --use_vae=False --use_highway=False --task=movie
    #   python3 train.py --use_checkpoint=False --use_mutual=False --use_vae=False --use_highway=False --task=movie --kl_anneal=True
    main()
