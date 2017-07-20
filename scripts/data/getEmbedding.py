#!/usr/bin/python
#-*- coding: utf-8 -*-

from os.path import dirname, abspath, join
import numpy as np
import cPickle as pickle
import ConfigParser
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
from util import get_cfg_path

def main():
    word_vec_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                        'data',
                        'word_vec')
    embedding_path = join(word_vec_dir, 'embeddings.txt')
    words_lst_path = join(word_vec_dir, 'words.lst')

    word2Idx, line_idx = {}, 0
    with open(words_lst_path, 'r') as f:
        for line in f:
            word2Idx[line[:-1]] = line_idx + 1
            line_idx += 1
    wordNum = line_idx + 1
    print('Word list size is {}'.format(line_idx + 1))

    cf = ConfigParser.ConfigParser()
    cf.read(get_cfg_path())
    cf.set('Data', 'word_list_len', line_idx + 1)

    line_idx = 0
    with open(embedding_path, 'r') as f:
        for line in f:
            if line_idx == 0:
                dim = len(line.split())
                print('Word vector dimension is {}'.format(dim))
                cf.set('Data', 'word_dim', dim)
                with open(get_cfg_path(), 'w') as f:
                    cf.write(f)
                word2vec = np.zeros((wordNum + 1, dim))

            line_str = line.split()
            word_vec = np.zeros((1, dim))
            for i in range(dim):
                word_vec[0, i] = float(line_str[i])
            word2vec[line_idx + 1, :] = word_vec
            line_idx += 1

    pickle.dump(word2vec,
                open(join(word_vec_dir, 'word2vec.dat'), 'wb'),
                True)
    pickle.dump(word2Idx,
                open(join(word_vec_dir, 'word2Idx.dat'), "wb"),
                True)
    print('change format done!, save file in word2vec.dat and word2Idx.dat')

if __name__ == '__main__':
    main()
