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

sys.path.append(dirname(dirname(abspath(__file__))))
from data.decode_tfrecords import read_and_decode
from util import get_num_records, get_cfg_path


class My_model(object):
    """ LSTM based DST model. """

    def __init__(self,
                 data,
                 mode,
                 lstm_units,
                 hidden_units,
                 output_units,
                 init,
                 beta=0.0001,
                 keep_prob=0.7):
        self.data = data
        self.mode = mode
        self.lstm_units = lstm_units
        # a list lstm-attention-fc-output
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.init = init
        self.beta = beta
        self.keep_prob = keep_prob
        self.regularizer = layers.l2_regularizer(scale=self.beta)

        with tf.name_scope('RNN'):
            lstm_outputs = self._rnn(data.input)

        with tf.name_scope('Attention'):
            attention_output = self._attention(lstm_outputs)

        pred = self._full_connected(attention_output)
        self.pred = nn.softmax(pred)
        self.pred_label = tf.argmax(pred, 1)

        if self.mode != 'test':
            with tf.name_scope('Label'):
                label = tf.identity(data.label, 'label')

            with tf.name_scope('Loss'):
                normal_loss = tf.reduce_mean(
                        nn.softmax_cross_entropy_with_logits(logits=pred,
                                                             labels=label))
                # it's a list
                reg_losses = tf.get_collection(GraphKeys.REGULARIZATION_LOSSES)
                loss = tf.add(normal_loss,
                              tf.add_n(reg_losses),
                              name='loss')
            self.loss = loss
            tf.summary.scalar('loss', self.loss)

            with tf.name_scope('Acc'):
                correct_prediction = tf.equal(tf.argmax(pred, 1),
                                                tf.argmax(label, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                    tf.float32),
                                            name='acc')
            self.accuracy = accuracy
            tf.summary.scalar('accuracy', self.accuracy)

    def _rnn(self, x):
        word_idx = x
        word_idx = tf.cast(word_idx, tf.int32)
        init_embedding = tf.constant(WordVec.word_vecs)
        with tf.variable_scope('Embeddings'):
            embeddings = tf.get_variable('embeddings',
                                         initializer=init_embedding)
        word_vec = nn.embedding_lookup(embeddings, word_idx)
        input = tf.unstack(word_vec, axis=1)

        lstm_cell = rnn.LSTMCell(self.lstm_units,
                                 use_peepholes=True,
                                 forget_bias=1.0,
                                 initializer=self.init)

        if self.mode == 'train':
            keep_prob = self.keep_prob
        else:
            keep_prob = 1.0

        lstm_cell = nn.rnn_cell.DropoutWrapper(lstm_cell,
                                               input_keep_prob=keep_prob,
                                               output_keep_prob=keep_prob)

        outputs, states = rnn.static_rnn(lstm_cell, input, dtype=tf.float32)

        return outputs

    def _attention(self, lstm_outputs):
        with tf.variable_scope('Attention', initializer=self.init):
            weights = tf.get_variable('weights',
                                      [self.lstm_units, self.output_units],
                                      regularizer=self.regularizer)
            biases = tf.get_variable('biases', [1, self.output_units])
            u_w = tf.get_variable('u_w', [self.output_units, 1])

        outputs, scores = [], []
        for v in lstm_outputs:
            hidden_rep = nn.tanh(tf.add(tf.matmul(v, weights), biases))
            scores.append(tf.matmul(hidden_rep, u_w))
        # list -> tensor
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

    def _full_connected(self, attention_output):
        hu = self.hidden_units
        for i in xrange(len(hu) + 1):
            with tf.variable_scope('Affine' + str(i + 1),
                                   initializer=self.init):
                if i == 0:
                    weights = tf.get_variable('weights',
                                              [self.lstm_units, hu[i]],
                                              regularizer=self.regularizer)
                    biases = tf.get_variable('biases', [hu[i]])
                    hidden_output = tf.add(tf.matmul(attention_output, weights),
                                           biases,
                                           name='hidden_output')
                elif i == len(hu):
                    weights = tf.get_variable('weights',
                                              [hu[i - 1], self.output_units],
                                              regularizer=self.regularizer)
                    biases = tf.get_variable('biases', [self.output_units])
                    with tf.name_scope('Pred'):
                        pred = tf.add(tf.matmul(hidden_output, weights),
                                      biases,
                                      name='pred')
                    return pred
                else:
                    weights = tf.get_variable('weights',
                                              [hu[i - 1], hu[i]],
                                              regularizer=self.regularizer)
                    biases = tf.get_variable('biases', [hu[i]])
                    hidden_output = tf.add(tf.matmul(hidden_output, weights),
                                           biases,
                                           name='hidden_output')


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
