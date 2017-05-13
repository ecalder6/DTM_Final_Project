from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Reader import Reader
from LSTMVAE import LSTMVAE

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
    model = LSTMVAE(lines, \
                args.batch_size, args.emb_size, \
                args.latent_size, reader.vocab_size, reader.max_length, \
                use_vae=args.use_vae, use_highway=args.use_highway, mutual_lambda=args.mutual_loss_lambda)
    out = model.get_outputs(replies)

    loss = model.get_loss(replies, out, use_mutual=args.use_mutual)
    kld = None
    z = None
    mutual_loss = None
    if args.use_vae:
        z = model.get_z()
        kld = model.get_kl()
    if args.use_mutual:
        mutual_loss = model.get_mutual_loss()
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
    mutual_losses = []
    zs = []
    duration = time.time()

    output_csv = args.task + "_" + str(args.iterations)
    if args.use_mutual:
        output_csv = output_csv + "_m"
    f = open(args.data_dir+"analytics/" + output_csv + "train.csv", "w", newline='')
    writer = csv.writer(f)
    z_file = None
    if args.use_mutual or args.use_vae:
        cov_file = open(args.data_dir+"analytics/cov.txt", "wb")
        if args.save_z:
            z_file = open(args.data_dir+"analytics/z", "wb")
            ordered_z = []
    anneal = model.get_anneal_assignment(1)

    for step in range(args.iterations):
        obj_l, kl_l, m_l, cov = None, None, None, None
        # Add kl annealing
        if args.kl_anneal:
            anneal = model.get_anneal_assignment(1/(1+(1.01)**(-step+999)))
        # Run one iteration for training and save the loss
        if args.use_mutual:
            _, obj_l, kl_l, m_l, cov = sess.run([train_op, loss, kld, mutual_loss, model._z_cov], {
                lr: learning_rate
            })
            kl_loss.append(kl_l)
            mutual_losses.append(m_l)
        elif args.use_vae:
            _, obj_l, kl_l, cov, kl_anneal = sess.run([train_op, loss, kld, model._z_cov, anneal], {
                lr: learning_rate
            })
            kl_loss.append(kl_l)
        else:
            _, obj_l = sess.run([train_op, loss], {
                lr: learning_rate
            })
        objective_loss.append(obj_l)

        if args.use_mutual or args.use_vae:
            np.savetxt(cov_file, cov[:5, :5])
            cov_file.write(str.encode("================================="))

        if step % args.update_every == 0:
            print("\nIteration: ", step+1)
            print("Duration: ", time.time()-duration )
            print("Objective loss: %.3f" % obj_l)
            if args.use_vae:
                print("KL loss: %.5f\n" % kl_l)

            if args.save_z:
                c, s, r, za = sess.run([lines, replies, sample, z])
                # if not len(ordered_z):
                #     ordered_z = c
                #     print(len(ordered_z))
                #     zs.append(za)
                # else:
                #     keys = map(lambda k: tuple(k), c)
                #     zdict = dict(zip(keys, za))
                #     print(zdict.keys())
                #     print(len(c))
                #     print(len(zdict.keys()))
                #     new_z = []
                #     for key in ordered_z:
                #         new_z.append(zdict[tuple(key)])
                #     zs.append(new_z)
                #     print(zs)
                zs.append(za)
                # print("exit")
                # exit()
            else:
                c, s, r = sess.run([lines, replies, sample])
            for i in range(5):
                print("====================================================")
                # Windows: chcp 65001
                print(to_eng(c[i], reader.meta['idx2w']), "-->", to_eng(r[:, i], reader.meta['idx2w']))
                print("True reply: ", to_eng(s[i], reader.meta['idx2w']))
                print("====================================================")
                # print(c[i])
                # print(r[:,i])

            if len(mutual_losses):
                writer.writerow(['Objective loss', 'KLD', 'Mutual loss'])
                writer.writerows(zip(objective_loss, kl_loss, mutual_losses))
            if len(kl_loss):
                writer.writerow(['Objective loss', 'KLD'])
                writer.writerows(zip(objective_loss, kl_loss))
            else:
                writer.writerow(['Objective loss'])
                writer.writerows(zip(objective_loss))


            # c, s, r = sess.run([tf.constant([[3., 3.]]), tf.constant([[3., 3.]]), sample])
            # l_ave = b_ave = d_ave = 0
            if args.use_checkpoint:
                saver.save(sess, args.checkpoint_path + "checkpoint", global_step=0)
            # tf.add_to_collection('vars', reader.batch_size)
            # print("saved")
            # exit()
            

    if args.save_z:
        np.savez(z_file, *zs)

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
    parser.add_argument('--latent_size', default=128, type=int)
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
