#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import sys
from os.path import abspath, dirname, join
import pandas as pd
from pandas import DataFrame

import tensorflow as tf

sys.path.append(join(dirname(abspath(__file__)), 'scripts'))
from data.decode_tfrecords import read_and_decode
from model.my_model import My_model, Input
from util import get_num_records, get_cfg_path

cf = ConfigParser.ConfigParser()
cf.read(get_cfg_path())
learning_rate = cf.getfloat('Model', 'learning_rate')
lstm_units = cf.getint('Model', 'lstm_units')
cnn_flag = cf.getint('Model', 'cnn_flag')
cnn_kernels = cf.getint('Model', 'cnn_kernels')
cnn_ngrams = cf.get('Model', 'cnn_ngrams')
cnn_ngrams = [int(x) for x in cnn_ngrams.split(',')]
rnn_flag = cf.getint('Model', 'rnn_flag')
hidden_units = cf.get('Model', 'hidden_units')
if hidden_units == '0':
    hidden_units = []
else:
    hidden_units=[int(x) for x in hidden_units.split(',')]
output_units = cf.getint('Model', 'output_units')
beta = cf.getfloat('Model', 'beta')
keep_prob = cf.getfloat('Model', 'keep_prob')
save_path = sys.argv[1]


with tf.device('/gpu:0'):
    with tf.name_scope('Test'):
        with tf.name_scope('TestInput'):
            test_data = Input('test')
        with tf.variable_scope('Model', reuse=None):
            test_model = My_model(data = test_data,
                                  mode='test',
                                  lstm_units=lstm_units,
                                  cnn_flag=cnn_flag,
                                  cnn_kernels=cnn_kernels,
                                  cnn_ngrams=cnn_ngrams,
                                  rnn_flag=rnn_flag,
                                  hidden_units=hidden_units,
                                  output_units=output_units,
                                  beta=beta,
                                  keep_prob=keep_prob)
saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver.restore(sess, join(save_path, 'model.ckpt'))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

test_Sentiment = sess.run(test_model.pred_label)
test_PhraseId = sess.run(test_data.PhraseId)

submission = pd.DataFrame({'PhraseId':test_PhraseId,
                           'Sentiment':test_Sentiment})
submission.to_csv(save_path + '.csv', index=False)

coord.request_stop()
coord.join(threads)

sess.close()

print('The ' + save_path + '.csv ' + 'file is generated!')




