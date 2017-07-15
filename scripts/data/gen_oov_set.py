#!/usr/bin/python
#-*- coding: utf-8 -*-

import cPickle as pickle
from os.path import dirname, abspath, join
import sys

def main():
    word2Idx_path = join(dirname(dirname(dirname(abspath(__file__)))),
                          'data',
                          'word_vec',
                          'word2Idx.dat')
    with open(word2Idx_path, 'r') as f:
        word2Idx = pickle.load(f)

    pickle_data_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                        'data',
                        'pickle_data')
    file_path = join(pickle_data_dir, sys.argv[1] + '_word_set.dat')
    with open(file_path, 'r') as f:
        word_set = pickle.load(f)

    oov_set = set()
    pickle.dump(oov_set,
                open(join(pickle_data_dir, sys.argv[1] + '_oov_set.dat'), "wb"),
                True)

    print('OOV set for ' + sys.argv[1] ' dataset is:')
    print(oov_set)

if __name__ == '__main__':
    main()
