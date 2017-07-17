#!/usr/bin/python
#-*- coding: utf-8 -*-

from os.path import abspath, dirname, join
import re
import tensorflow as tf

def get_cfg_path():
    config_dir = join(dirname(dirname(abspath(__file__))), 'config')
    cfg_path = join(config_dir, 'config.cfg')
    return cfg_path

def get_num_records(tf_record_file):
    return len([x for x in tf.python_io.tf_record_iterator(tf_record_file)])

def get_file_num_line(file_path):
    num_line = 0
    with open(file_path, 'r') as f:
        for line in f:
            num_line += 1
    return num_line

def oov_word_proc(oov):
    # xxx-xxx
    if re.match('.+-.+', oov):
        return oov.split('-')
    # number -> 0
    if re.match('\d+', oov):
        return ['0']
    # xxxly -> xxx
    if re.match('.+ly', oov):
        return [oov[:-2]]
    # xxxtted -> xxxt
    if re.match('.+tted', oov):
        return [oov[:-3]]
    # xxxed -> xxx
    if re.match('.+ed', oov):
        return [oov[:-2]]
    # unxxx -> not xxx
    if re.match('un.+', oov):
        return ['not', oov[2:]]
    # xxx\\/xxx
    if re.match('.+\\/.+', oov):
        return oov.split('\\/')
    # default
    return [oov]


def main():
    pass

if __name__ == '__main__':
    main()
