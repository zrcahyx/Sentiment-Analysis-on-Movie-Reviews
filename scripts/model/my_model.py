#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import cPickle as pickle
from os.path import abspath, dirname, join
import sys

import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow import GraphKeys
from tensorflow.contrib.layers import xavier_initializer
from tensorflow import zeros_initializer

sys.path.append(dirname(dirname(abspath(__file__))))
from data.decode_tfrecords import read_and_decode
from util import get_num_records, get_cfg_path


class My_model(object):
    """ LSTM based DST model. """

    def __init__(self,
                 data,
                 mode,
                 lstm_units,
                 cnn_flag,
                 cnn_kernels,
                 cnn_ngrams,
                 rnn_flag,
                 hidden_units,
                 output_units,
                 beta,
                 keep_prob):
        cf = ConfigParser.ConfigParser()
        cf.read(get_cfg_path())
        self.data = data
        self.mode = mode
        self.lstm_units = lstm_units
        self.cnn_kernels = cnn_kernels
        self.cnn_ngrams = cnn_ngrams
        self.cnn_flag = cnn_flag
        self.rnn_flag = rnn_flag
        # a list lstm-attention-fc-output
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.beta = beta
        self.keep_prob = keep_prob
        self.regularizer = layers.l2_regularizer(scale=1.0)
        self.seq_len = cf.getint('Data', self.mode + '_seq_len')

        with tf.name_scope('Look_up'):
            word_vec = self._look_up(data.input)

        if self.rnn_flag:
            with tf.name_scope('RNN_Attention'):
                rnn_attention_feature = self._rnn_attention(word_vec)

        if self.cnn_flag:
            with tf.name_scope('CNN'):
                cnn_feature = self._cnn(word_vec)

        if self.cnn_flag and self.rnn_flag:
            with tf.name_scope('ConFeatures'):
                features = tf.concat([rnn_attention_feature, cnn_feature],
                                    axis=1, name="features")
            # use average cnn feature
            self.feature_dim = (self.cnn_kernels + self.lstm_units)
        elif not self.cnn_flag:
            features = rnn_attention_feature
            self.feature_dim = self.lstm_units
        elif not self.rnn_flag:
            features = cnn_feature
            self.feature_dim = self.cnn_kernels
            # self.feature_dim = self.cnn_kernels * len(self.cnn_ngrams)

        if self.mode == 'train':
            keep_prob = self.keep_prob
        else:
            keep_prob = 1.0

        with tf.name_scope('Dropout'):
            features = nn.dropout(features, keep_prob=keep_prob,
                                  name='dropout')

        logits = self._full_connected(features)
        self.pred = nn.softmax(logits)
        self.pred_label = tf.argmax(logits, 1)

        if self.mode != 'test':
            with tf.name_scope('Label'):
                label = tf.identity(data.label, 'label')

            with tf.name_scope('Loss'):
                normal_loss = tf.reduce_mean(
                        nn.softmax_cross_entropy_with_logits(logits=logits,
                                                             labels=label))
                # it's a list
                reg_losses = tf.get_collection(GraphKeys.REGULARIZATION_LOSSES)
                loss = tf.add(normal_loss,
                              tf.add_n(reg_losses) * self.beta,
                              name='loss')
            self.loss = loss
            tf.summary.scalar('loss', self.loss)

            with tf.name_scope('Acc'):
                correct_prediction = tf.equal(tf.argmax(logits, 1),
                                                tf.argmax(label, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                    tf.float32),
                                            name='acc')
            self.accuracy = accuracy
            tf.summary.scalar('accuracy', self.accuracy)

    def _look_up(self, x):
        word_idx = x
        word_idx = tf.cast(word_idx, tf.int32)
        init_embedding = tf.constant(WordVec.word_vecs)
        with tf.variable_scope('Embeddings'):
            embeddings = tf.get_variable('embeddings',
                                         initializer=init_embedding)
        word_vec = nn.embedding_lookup(embeddings, word_idx)
        return word_vec

    def _rnn_attention(self, word_vec):
        input = tf.unstack(word_vec, axis=1)

        with tf.variable('LSTM'):
            lstm_cell = rnn.LSTMCell(
                            self.lstm_units,
                            use_peepholes=True,
                            forget_bias=1.0,
                            initializer=tf.random_uniform_initializer(-0.1, 0.1))

        if self.mode == 'train':
            keep_prob = self.keep_prob
        else:
            keep_prob = 1.0

        lstm_cell = nn.rnn_cell.DropoutWrapper(lstm_cell,
                                               input_keep_prob=keep_prob,
                                               output_keep_prob=keep_prob)

        lstm_outputs, states = rnn.static_rnn(lstm_cell, input, dtype=tf.float32)

        with tf.variable_scope('Attention'):
            weights = tf.get_variable(
                        'weights',
                        [self.lstm_units, self.output_units],
                        regularizer=self.regularizer,
                        initializer=xavier_initializer(uniform=False))
            biases = tf.get_variable('biases',
                                     [1, self.output_units],
                                     initializer=zeros_initializer())
            u_w = tf.get_variable('u_w', [self.output_units, 1])

        outputs, scores = [], []
        for v in lstm_outputs:
            hidden_rep = nn.tanh(tf.add(tf.matmul(v, weights), biases))
            scores.append(tf.matmul(hidden_rep, u_w))
        # list -> tensor batch_size x seq_len
        scores = tf.concat(scores, axis=1)
        # softmax
        scores = nn.softmax(scores, dim=-1)
        # tensor -> list
        scores = tf.unstack(scores, axis=1)

        for i, v in enumerate(scores):
            # v: (64,) -> (64,1)
            v = tf.reshape(v, [-1, 1])
            # v: (64,1) -> [(64,1), (64,1), ...]
            v = [v] * self.lstm_units
            # v: (64,self.lstm_units)
            v = tf.concat(v, axis=1)
            outputs.append(tf.multiply(v, lstm_outputs[i]))

        return tf.add_n(outputs)

    def _cnn(self, word_vec):
        cf = ConfigParser.ConfigParser()
        cf.read(get_cfg_path())
        word_dim = cf.getint('Data', 'word_dim')
        # NHWC format
        word_vec = tf.reshape(word_vec, [-1, self.seq_len, word_dim, 1])
        cnn_feature = []
        for i, v in enumerate(self.cnn_ngrams):
            with tf.variable_scope('conv_{}'.format(v)):
                weights = tf.get_variable(
                                'weights',
                                [v, word_dim, 1, self.cnn_kernels],
                                regularizer=None,
                                initializer=xavier_initializer(uniform=False))
                biases = tf.get_variable('biases',
                                         [self.cnn_kernels],
                                         initializer=zeros_initializer())
            # batch_size x H x 1 x kernels
            conv = nn.conv2d(word_vec, weights,
                             strides=[1,1,1,1],padding='VALID')
            # batch_size x H x 1 x kernels
            conv_relu = nn.relu(conv + biases)
            H = self.seq_len - v + 1
            # batch_size x 1 x 1 x kernels
            conv_relu_pooling = nn.max_pool(conv_relu, ksize=[1, H, 1, 1],
                                            strides=[1,1,1,1], padding='VALID')
            feature = tf.reshape(conv_relu_pooling, [-1, self.cnn_kernels])
            cnn_feature.append(feature)
        # use average cnn feature
        cnn_feature = tf.divide(tf.add_n(cnn_feature), len(cnn_feature))
        # cnn_feature = tf.concat(cnn_feature, axis=1)
        return cnn_feature

    def _full_connected(self, features):
        if self.hidden_units:
            hu = self.hidden_units
            for i in xrange(len(hu) + 1):
                with tf.variable_scope('Affine' + str(i + 1)):
                    if i == 0:
                        weights = tf.get_variable(
                                    'weights',
                                    [self.feature_dim, hu[i]],
                                    regularizer=self.regularizer,
                                    initializer=xavier_initializer(uniform=False))
                        biases = tf.get_variable(
                                    'biases',
                                    [hu[i]],
                                    initializer=zeros_initializer())
                        hidden_output = tf.add(
                                            tf.matmul(features, weights),
                                            biases,
                                            name='hidden_output')
                    elif i == len(hu):
                        weights = tf.get_variable(
                                    'weights',
                                    [hu[i - 1], self.output_units],
                                    regularizer=self.regularizer,
                                    initializer=xavier_initializer(uniform=False))
                        biases = tf.get_variable('biases',
                                                 [self.output_units],
                                                 initializer=zeros_initializer())
                        with tf.name_scope('Logits'):
                            logits = tf.add(tf.matmul(hidden_output, weights),
                                        biases,
                                        name='logits')
                        return logits
                    else:
                        weights = tf.get_variable(
                                    'weights',
                                    [hu[i - 1], hu[i]],
                                    regularizer=self.regularizer,
                                    initializer=xavier_initializer(uniform=False))
                        biases = tf.get_variable('biases',
                                                [hu[i]],
                                                initializer=zeros_initializer())
                        hidden_output = tf.add(tf.matmul(hidden_output, weights),
                                            biases,
                                            name='hidden_output')
        elif not self.hidden_units:
            with tf.variable_scope('Affine'):
                weights = tf.get_variable(
                            'weights',
                            [self.feature_dim, self.output_units],
                            regularizer=self.regularizer,
                            initializer=xavier_initializer(uniform=False))
                biases = tf.get_variable('biases',
                                         [self.output_units],
                                         initializer=zeros_initializer())
            with tf.name_scope('Logits'):
                logits = tf.add(tf.matmul(features, weights),
                              biases,
                              name='logits')
            return logits


class WordVec(object):
    _word_weights_path = join(dirname(dirname(dirname(abspath(__file__)))),
                              'data',
                              'word_vec',
                              'word2vec.dat')
    with open(_word_weights_path, 'r') as f:
        word_vecs = pickle.load(f).astype(np.float32)


class Input(object):
    """ The input data. """

    def __init__(self, mode):
        if mode == 'test':
            input, PhraseId, records_num = self._get_data(mode)
            self.input = input
            self.PhraseId = PhraseId
            self.records_num = records_num
        else:
            input, label, records_num = self._get_data(mode)
            self.input = input
            self.label = label
            self.records_num = records_num

    @staticmethod
    def _get_data(mode):
        cf = ConfigParser.ConfigParser()
        cf.read(get_cfg_path())
        output_units = cf.getint('Model', 'output_units')
        batch_size = cf.getint('Model', 'batch_size')

        filename = join(dirname(dirname(dirname(abspath(__file__)))),
                        'data',
                        'tfrecords_data',
                        mode + '.tfrecords')

        num_epochs = None
        records_num = get_num_records(filename)
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=num_epochs)
        if mode == 'test':
            input, PhraseId, _ = read_and_decode(filename_queue, mode)
            batch_size = records_num
            input, PhraseId = tf.train.batch([input, PhraseId],
                                   batch_size=batch_size,
                                   num_threads=2,
                                   capacity=1000 + 3 * batch_size)
            return input, PhraseId, records_num
        elif mode == 'dev':
            input, label, _, _ = read_and_decode(filename_queue, mode)
            label = tf.one_hot(label, output_units, name='label')
            batch_size = records_num
            input, label = tf.train.batch([input, label],
                                          batch_size=batch_size,
                                          num_threads=2,
                                          capacity=1000 + 3 * batch_size)
            return input, label, records_num
        elif mode == 'train':
            input, label, _, _ = read_and_decode(filename_queue, mode)
            label = tf.one_hot(label, output_units, name='label')
            batch_size = batch_size
            input, label = tf.train.shuffle_batch([input, label],
                                                  batch_size=batch_size,
                                                  num_threads=2,
                                                  capacity=1000 + 3 * batch_size,
                                                  min_after_dequeue=1000)
            return input, label, records_num
