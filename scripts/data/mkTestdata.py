#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from os.path import dirname, abspath, join
import sys
import ConfigParser
import cPickle as pickle
import re

sys.path.append(dirname(dirname(abspath(__file__))))
from util import get_cfg_path, get_file_num_line, oov_word_proc

def main():
    word2Idx_path = join(dirname(dirname(dirname(abspath(__file__)))),
                          'data',
                          'word_vec',
                          'word2Idx.dat')
    with open(word2Idx_path, 'r') as f:
        word2Idx = pickle.load(f)

    test_data, test_word_set, test_oov_set = [], set(), set()
    test_num_oov = 0
    tsv_data_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                        'data',
                        'tsv_data')
    file_path = join(tsv_data_dir, 'test.tsv')

    test_examples = get_file_num_line(file_path) - 1
    print('Number of phrase for test dataset is {}'.format(test_examples))
    cf = ConfigParser.ConfigParser()
    cf.read(get_cfg_path())
    cf.set('Data', 'test_num_phrase', test_examples)

    line_idx = 0
    with open(file_path, 'r') as f:
        for line in f:
            example = {}
            if line_idx == 0:
                line_idx += 1
                continue

            line_str = line.split('\t')
            example['PhraseId'] = int(line_str[0])
            example['SentenceId'] = int(line_str[1])
            input_raw = line_str[2].lower()

            input_str_raw = input_raw.split()
            input_idx, input_str = [], []
            for v in input_str_raw:
                if v in word2Idx.keys():
                    input_idx.append(word2Idx[v])
                    input_str.append(v)
                else:
                    v2list = oov_word_proc(v)
                    for vv in v2list:
                        input_str.append(vv)
                        if vv in word2Idx.keys():
                            input_idx.append(word2Idx[vv])
                        else:
                            input_idx.append(word2Idx['UNKNOWN'])
            example['input'] = np.array(input_idx)

            test_data.append(example)
            for v in input_str:
                test_word_set.add(v)

            print('Line {} processing is done!'.format(line_idx))
            line_idx += 1

    test_num_sentence = (test_data[-1]['SentenceId'] -
                            test_data[0]['SentenceId'] +
                            1)
    print('Number of sentences for test dataset is {}'.format(test_num_sentence))
    cf.set('Data', 'test_num_sentence', test_num_sentence)

    print('Number of different words for test dataset is {}'.format(len(test_word_set)))
    cf.set('Data', 'test_num_word', len(test_word_set))

    for v in list(test_word_set):
        if not v in word2Idx.keys():
            test_num_oov += 1
            test_oov_set.add(v)

    print('Number of oov words for test dataset is {}'.format(test_num_oov))
    cf.set('Data', 'test_num_oov', test_num_oov)

    with open(get_cfg_path(), 'w') as f:
        cf.write(f)

    pickle_data_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                          'data',
                          'pickle_data')
    pickle.dump(test_data,
                open(join(pickle_data_dir, 'test.dat'), "wb"),
                True)
    pickle.dump(test_word_set,
                open(join(pickle_data_dir, 'test_word_set.dat'), "wb"),
                True)
    pickle.dump(test_oov_set,
                open(join(pickle_data_dir, 'test_oov_set.dat'), "wb"),
                True)

    print('OOV set for test dataset is:')
    print(test_oov_set)

if __name__ == '__main__':
    main()
