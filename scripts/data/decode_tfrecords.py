#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import sys
from os.path import abspath, dirname

import tensorflow as tf

sys.path.append(dirname(dirname(abspath(__file__))))
from util import get_cfg_path


def read_and_decode(filename_queue, mode):
    cf = ConfigParser.ConfigParser()
    cf.read(get_cfg_path())
    seq_len = cf.getint('Data', mode + '_seq_len')

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件

    if mode == 'test':
        features = tf.parse_single_example(
            serialized_example,
            features={
                # tf.FixedLenFeature: Configuration for parsing a fixed-length input feature.
                'input': tf.FixedLenFeature([], tf.string),
                'PhraseId': tf.FixedLenFeature([], tf.int64),
                'SentenceId': tf.FixedLenFeature([], tf.int64)
            })
        # tf.decode_raw: Reinterpret the bytes of a string as a vector of numbers.
        input = tf.decode_raw(features['input'], tf.float32)
        input.set_shape([seq_len])
        PhraseId = features['PhraseId']
        SentenceId = features['SentenceId']
        return input, PhraseId, SentenceId
    else:
        features = tf.parse_single_example(
        serialized_example,
        features={
            # tf.FixedLenFeature: Configuration for parsing a fixed-length input feature.
            'input': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'PhraseId': tf.FixedLenFeature([], tf.int64),
            'SentenceId': tf.FixedLenFeature([], tf.int64)
        })
        # tf.decode_raw: Reinterpret the bytes of a string as a vector of numbers.
        input = tf.decode_raw(features['input'], tf.float32)
        input.set_shape([seq_len])
        label = features['label']
        PhraseId = features['PhraseId']
        SentenceId = features['SentenceId']
        return input, label, PhraseId, SentenceId
