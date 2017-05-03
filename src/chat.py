from LSTMVAE import LSTMVAE
import tensorflow as tf
import argparse
from Reader import Reader

def to_eng(ids, ix_to_word):
    output = ""
    for id in ids:
        if id:
            output += ix_to_word[id] + " "
    return output

def main():
    args = get_args()

    sess = tf.Session()
    saver = tf.train.import_meta_graph(args.data_dir + args.checkpoint_path + args.checkpoint_meta_file)
    checkpoint_path = args.checkpoint_path
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--checkpoint_path', default='../twitter_checkpoint_vae_highway/', type=str)
    parser.add_argument('--checkpoint_meta_file', default='checkpoint-0.meta', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
