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
    reader = Reader(data_dir=args.data_dir)
    reader.read_metadata()
    tweets, replies = reader.read_records()
    learning_rate = args.learning_rate


    # Set up model
    model = LSTMVAE(tweets, \
                reader.batch_size, args.emb_size, \
                args.latent_size, reader.vocab_size, reader.max_length, \
                use_vae=args.use_vae, use_highway=args.use_highway)
    out = model.get_outputs(replies)

    loss = model.get_loss(replies, out, use_mutual=args.use_mutual)
    kld = None
    if args.use_vae:
        kld = model.get_kl()
    lr = tf.placeholder_with_default(learning_rate, [], name="lr")
    train_op = model.train(lr, loss)
    sample = model.sample(args.sample_temp)


    # Set up training session
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Set up checkpoint
    if args.use_checkpoint:
        saver = tf.train.Saver()
        checkpoint_path = args.checkpoint_path
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)

    ### Main loop for training
    l_ave = b_ave = d_ave = 0
    objective_loss = []
    kl_loss = []
    duration = time.time()
    for step in range(args.iterations):
        obj_l = kl_l = None
        if args.use_vae:
            # Run one iteration for training and save the loss
            _, obj_l, kl_l = sess.run([train_op, loss, kld], {
                lr: learning_rate
            })
            kl_loss.append(kl_l)
        else:
            _, obj_l = sess.run([train_op, loss], {
                lr: learning_rate
            })
        objective_loss.append(obj_l)


        if step % args.update_every == 0:
            print("\nIteration: ", step+1)
            print("Duration: ", time.time()-duration )
            print("Objective loss: %.3f" % obj_l)
            if args.use_vae:
                print("KL loss: %.5f\n" % kl_l)

            c, s, r = sess.run([tweets, replies, sample])
            for i in range(5):
                print("====================================================")
                # Windows: chcp 65001
                print(to_eng(c[i], reader.meta['idx2w']), "-->", to_eng(r[:, i], reader.meta['idx2w']))
                print("True reply: ", to_eng(s[i], reader.meta['idx2w']))
                print("====================================================")
                print(c[i])
                print(r[:,i])
                exit()

            l_ave = b_ave = d_ave = 0
            saver.save(sess, args.checkpoint_path, global_step=0)
    output_csv = args.task + "_" + str(args.iterations)
    if args.use_mutual:
        output_csv = output_csv + "_m"
    with open(args.data_dir+"analytics/" + output_csv + "train.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Objective loss', 'KLD'])
        writer.writerows(zip(objective_loss, kl_loss))
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
    parser.add_argument('--learning_rate', default=0.01, type=int)
    parser.add_argument('--rnn_hidden_size', default=512, type=int)
    parser.add_argument('--emb_size', default=512, type=int)
    parser.add_argument('--keep_prob', default=0.8, type=float)
    parser.add_argument('--sample_temp', default=0.7, type=float)
    parser.add_argument('--latent_size', default=128, type=int)
    parser.add_argument('--iterations', default=5000, type=int)
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--checkpoint_path', default='../twitter_checkpoint/', type=str)
    parser.add_argument('--run_converter', default=False, type=str2bool)
    parser.add_argument('--use_mutual', default=False, type=str2bool)
    parser.add_argument('--use_vae', default=True, type=str2bool)
    parser.add_argument('--use_highway', default=True, type=str2bool)
    parser.add_argument('--task', default="twitter", type=str)
    parser.add_argument('--update_every', default=100, type=int)
    parser.add_argument('--use_checkpoint', default=True, type=str2bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
