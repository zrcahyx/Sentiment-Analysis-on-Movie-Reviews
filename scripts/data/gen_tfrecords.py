#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import sys
from os.path import abspath, dirname, join

import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main():
    # train/dev/test
    filename = sys.argv[1]
    tf_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                  'data',
                  'tfrecords_data')
    writer = tf.python_io.TFRecordWriter(join(tf_dir, filename + ".tfrecords"))
    padding_dir = join(dirname(dirname(dirname(abspath(__file__)))),
                      'data',
                      'padding_data')
    with open(join(padding_dir, filename + '_padding.dat'), 'rb') as f:
        # this is a list
        dataset = pickle.load(f)

    for i, v in enumerate(dataset):
        # bytes格式
        input_raw = dataset[i]['input'].astype(np.float32).tobytes()
        session_id_raw = dataset[i]['session_id'].encode('utf-8')
        request_label_raw = (np.array(dataset[i]['output'][-1])
                             .astype(np.float32).tobytes())
        example = tf.train.Example(features=tf.train.Features(feature={
            'input': _bytes_feature(input_raw),
            'label': _bytes_feature(normal_label_raw),
            'request_label': _bytes_feature(request_label_raw),
            'session_id': _bytes_feature(session_id_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()
