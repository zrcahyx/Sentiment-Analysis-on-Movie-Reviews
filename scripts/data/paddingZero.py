#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import cPickle as pickle
import sys
from os.path import abspath, dirname, join

import numpy as np

sys.path.append(dirname(dirname(abspath(__file__))))
from util import get_cfg_path


def _zero_padding():
    pickle_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                      'data',
                      'pickle_data')
    filepath = join(pickle_dir, sys.argv[1] + '.dat')
    with open(filepath, 'rb') as f:
        # dataset is a list
        dataset = pickle.load(f)
    # record max length of turn input
    maxLen = 0

    for i in xrange(len(dataset)):
        # dataset[i]['input'] is a list
        maxLen = max(dataset[i]['input'].shape[0], maxLen)
    print('Max Length = ' + str(maxLen))  # MAX length

    cf = ConfigParser.ConfigParser()
    cf.read(get_cfg_path())
    cf.set('Data', sys.argv[1] + '_seq_len', maxLen)
    with open(get_cfg_path(), 'w') as f:
        cf.write(f)

    for i in xrange(len(dataset)):
        print('Processing example {}!'.format(i + 1))
        # each turn's old input len
        oldLen = dataset[i]['input'].shape[0]
        margin_Len = maxLen - oldLen
        margin_array = np.zeros(margin_Len)
        dataset[i]['input'] = np.concatenate((dataset[i]['input'], margin_array))

    padding_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                      'data',
                      'padding_data')
    pickle.dump(dataset,
                open(join(padding_dir, sys.argv[1] + "_padding.dat"), "wb"),
                True)

    print('Padding zero is done!')

def main():
    _zero_padding()


if __name__ == '__main__':
    main()
