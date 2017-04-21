from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Reader import Reader

import os, random, time, glob, pickle

import numpy as np
import tensorflow as tf


DATA_DIR = "./data/"

def to_eng(ids):
    return ' '.join([ix_to_word[id] if id != 0 else '' for id in ids])

def main():
    pass


if __name__ == "__main__":
    reader = Reader()

    main()