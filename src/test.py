from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Reader import Reader
from Converter import Converter
from LSTMVAE import LSTMVAE

import os, random, time, glob, pickle

import tensorflow as tf
import numpy as np

from functools import reduce
import operator
import argparse
import csv



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

    # if args.run_converter:
    #     converter = Converter(input_filename = args.data_dir + 'twitter.txt', output_filename = args.data_dir + 'twitter.tfrecords', meta_file = args.data_dir + 'metadata')
    #     converter.process_data()

    # Set up and run reader
    reader = Reader(data_dir=args.data_dir, task=args.task)
    reader.read_metadata()
    tweets, replies = reader.read_records(train=False)
    learning_rate = args.learning_rate


    # Set up model
    model = LSTMVAE(tweets, \
                reader.batch_size, args.emb_size, \
                args.latent_size, reader.vocab_size, reader.max_length, \
                use_vae=args.use_vae, use_highway=args.use_highway)

    # For some reason I have to call get_outputs and then model.sample. 
    # Just calling sample doesn't work. SAD
    output = model.get_outputs(replies)
    sample = model.sample(args.sample_temp)

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
    else:
        print("ERROR: checkpoint not found")
        exit()

    kld = model.get_kl()
    x, y, pred, kl_loss = sess.run([tweets, replies, sample, kld])

    loss = []
    for i in range(reader.batch_size):
        curr_loss = np.log(np.linalg.norm(np.subtract(np.array(y[i]), np.array(pred[:, i])))**2 + kl_loss)
        loss.append(curr_loss)
        print("====================================================")
        print(to_eng(x[i], reader.meta['idx2w']), "-->", to_eng(pred[:, i], reader.meta['idx2w']))
        print("True reply: ", to_eng(y[i], reader.meta['idx2w']))
        print("Loss: ", curr_loss)
        print("====================================================")




    output_csv = args.task + "_"
    if args.use_mutual:
        output_csv = output_csv + "_m"
    with open(args.data_dir+"./analytics/" + output_csv + "test.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(['loss','abg'])
        writer.writerows(loss)
    print("DONE")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=int)
    parser.add_argument('--rnn_hidden_size', default=512, type=int)
    parser.add_argument('--emb_size', default=512, type=int)
    parser.add_argument('--keep_prob', default=0.8, type=float)
    parser.add_argument('--sample_temp', default=0.7, type=float)
    parser.add_argument('--latent_size', default=128, type=int)
    parser.add_argument('--iterations', default=5000, type=int)
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--checkpoint_path', default='../checkpoints/', type=str)
    parser.add_argument('--run_converter', default=False, type=bool)
    parser.add_argument('--use_mutual', default=False, type=bool)
    parser.add_argument('--use_vae', default=True, type=bool)
    parser.add_argument('--use_highway', default=True, type=bool)
    parser.add_argument('--task', default="twitter", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
