#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from os.path import dirname, abspath, join
import sys
import ConfigParser
import cPickle as pickle

sys.path.append(dirname(dirname(abspath(__file__))))
from util import get_cfg_path, get_file_num_line

def main():
    word2Idx_path = join(dirname(dirname(dirname(abspath(__file__)))),
                          'data',
                          'word_vec',
                          'word2Idx.dat')
    with open(word2Idx_path, 'r') as f:
        word2Idx = pickle.load(f)

    train_data, dev_data, train_word_set, dev_word_set = [], [], set(), set()
    train_num_oov, dev_num_oov = 0, 0
    tsv_data_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                        'data',
                        'tsv_data')
    file_path = join(tsv_data_dir, 'train.tsv')

    total_examples = get_file_num_line(file_path) - 1
    train_examples = int(total_examples * 0.8)
    print('Number of phrase for train dataset is {}'.format(train_examples))
    dev_examples = total_examples - train_examples
    print('Number of phrase for dev dataset is {}'.format(dev_examples))
    cf = ConfigParser.ConfigParser()
    cf.read(get_cfg_path())
    cf.set('Data', 'train_num_phrase', train_examples)
    cf.set('Data', 'dev_num_phrase', dev_examples)

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
            example['label'] = int(line_str[3])

            input_str = input_raw.split()
            input_idx = []
            for v in input_str:
                if v in word2Idx.keys():
                    input_idx.append(word2Idx[v])
                else:
                    input_idx.append(word2Idx['UNKNOWN'])
            example['input'] = np.array(input_idx)

            if line_idx <= train_examples:
                train_data.append(example)
                for v in input_str:
                    train_word_set.add(v)
            else:
                dev_data.append(example)
                for v in input_str:
                    dev_word_set.add(v)
            print('Line {} processing is done!'.format(line_idx))
            line_idx += 1

    train_num_sentence = (train_data[-1]['SentenceId'] -
                            train_data[0]['SentenceId'] +
                            1)
    print('Number of sentences for train dataset is {}'.format(train_num_sentence))
    dev_num_sentence = (dev_data[-1]['SentenceId'] -
                            dev_data[0]['SentenceId'] +
                            1)
    print('Number of sentences for dev dataset is {}'.format(dev_num_sentence))
    cf.set('Data', 'train_num_sentence', train_num_sentence)
    cf.set('Data', 'dev_num_sentence', dev_num_sentence)

    print('Number of different words for train dataset is {}'.format(len(train_word_set)))
    print('Number of different words for dev dataset is {}'.format(len(dev_word_set)))
    cf.set('Data', 'train_num_word', len(train_word_set))
    cf.set('Data', 'dev_num_word', len(dev_word_set))

    for v in list(train_word_set):
        if not v in word2Idx.keys():
            train_num_oov += 1
    for v in list(dev_word_set):
        if not v in word2Idx.keys():
            dev_num_oov += 1
    print('Number of oov words for train dataset is {}'.format(train_num_oov))
    print('Number of oov words for dev dataset is {}'.format(dev_num_oov))
    cf.set('Data', 'train_num_oov', train_num_oov)
    cf.set('Data', 'dev_num_oov', dev_num_oov)

    with open(get_cfg_path(), 'w') as f:
        cf.write(f)

    picke_data_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                          'data',
                          'pickle_data')
    pickle.dump(train_data,
                open(join(picke_data_dir, 'train.dat'), "wb"),
                True)
    pickle.dump(dev_data,
                open(join(picke_data_dir, 'dev.dat'), "wb"),
                True)
    pickle.dump(train_word_set,
                open(join(picke_data_dir, 'train_word_set.dat'), "wb"),
                True)
    pickle.dump(dev_word_set,
                open(join(picke_data_dir, 'dev_word_set.dat'), "wb"),
                True)

if __name__ == '__main__':
    main()