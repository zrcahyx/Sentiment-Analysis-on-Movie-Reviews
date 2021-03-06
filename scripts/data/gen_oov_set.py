#!/usr/bin/python
#-*- coding: utf-8 -*-

import cPickle as pickle
from os.path import dirname, abspath, join
import sys
import ConfigParser

sys.path.append(dirname(dirname(abspath(__file__))))
from util import get_cfg_path

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

    oov_set, num_oov = set(), 0
    for v in list(word_set):
        if not v in word2Idx.keys():
            num_oov += 1
            oov_set.add(v)
            print('Word <' + v + '> is not in the word list!')
            print('Current number of oov word is: {}'.format(num_oov))

    cf = ConfigParser.ConfigParser()
    cf.read(get_cfg_path())
    cf.set('Data', sys.argv[1] + '_num_oov', num_oov)
    with open(get_cfg_path(), 'w') as f:
        cf.write(f)

    pickle.dump(oov_set,
                open(join(pickle_data_dir, sys.argv[1] + '_oov_set.dat'), "wb"),
                True)

    print('OOV set for ' + sys.argv[1] + ' dataset is:')
    print(oov_set)
    print('Number of OOV word in ' + sys.argv[1] + ' dataset is: {}'.format(num_oov))

if __name__ == '__main__':
    main()
