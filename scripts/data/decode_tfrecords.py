#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import sys
from os.path import abspath, dirname

import tensorflow as tf

sys.path.append(dirname(dirname(abspath(__file__))))
from util.cfg_helper import get_cfg_path


def read_and_decode(filename_queue, seq_len):
    cf = ConfigParser.ConfigParser()
    cf.read(get_cfg_path())

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件

    features = tf.parse_single_example(
        serialized_example,
        features={
            # tf.FixedLenFeature: Configuration for parsing a fixed-length input feature.
            'input': tf.FixedLenFeature([], tf.string),
            'normal_label': tf.FixedLenFeature([], tf.string),
            'request_label': tf.FixedLenFeature([], tf.string),
            'session_id': tf.FixedLenFeature([], tf.string),
        })
    # tf.decode_raw: Reinterpret the bytes of a string as a vector of numbers.
    turn_input = tf.decode_raw(features['input'], tf.float32)
    turn_input.set_shape([seq_len * 2])
    turn_input = tf.reshape(turn_input, [-1, 2])
    normal_label = tf.decode_raw(features['normal_label'], tf.int32)
    normal_label.set_shape([cf.getint('Data', 'num_normal_slot')])
    request_label = tf.decode_raw(features['request_label'], tf.float32)
    request_label.set_shape([cf.getint('Data', 'num_requested_slot')])
    session_id = features['session_id']
    return turn_input, normal_label, request_label, session_id
