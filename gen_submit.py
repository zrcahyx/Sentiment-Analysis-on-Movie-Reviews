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
from model.lstm_model import LSTM_attention, Input
from util import get_num_records, get_cfg_path

cf = ConfigParser.ConfigParser()
cf.read(get_cfg_path())
learning_rate = cf.getfloat('Model', 'learning_rate')
lstm_units = cf.getint('Model', 'lstm_units')
output_units = cf.getint('Model', 'output_units')
beta = cf.getfloat('Model', 'beta')
keep_prob = cf.getfloat('Model', 'keep_prob')
save_path = cf.get('Model', 'save_path')

init = tf.random_uniform_initializer(-0.1, 0.1)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

with tf.device('/gpu:0'):
    with tf.name_scope('Test'):
        with tf.name_scope('TestInput'):
            test_data = Input('test')
        with tf.variable_scope('Model', reuse=None):
            test_model = LSTM_attention(data = test_data,
                                        mode='test',
                                        lstm_units=lstm_units,
                                        output_units=output_units,
                                        opt = opt,
                                        init = init,
                                        beta=beta,
                                        keep_prob=keep_prob)
saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver.restore(sess, 'saved_model/model.ckpt')

test_Sentiment = sess.run(test_model.pred_label)
test_PhraseId = test_data.PhraseId

submission = pd.DataFrame({'PhraseId':test_PhraseId,
                           'Sentiment':test_Sentiment})
submission.to_csv('submission.csv', index=False)




