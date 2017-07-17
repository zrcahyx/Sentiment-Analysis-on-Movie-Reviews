#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
from os.path import abspath, dirname, join

import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import rnn


class LSTM_attention(object):
    """ LSTM based DST model. """

    def __init__(self,
                 data,
                 mode,
                 lstm_units,
                 output_units,
                 opt,
                 init,
                 beta=0.0001,
                 keep_prob=0.7):
        self.data = data
        self.mode = mode
        self.lstm_units = lstm_units
        self.output_units = output_units
        self.opt = opt
        self.init = init
        self.beta = beta
        self.keep_prob = keep_prob

        with tf.name_scope('RNN'):
            lstm_outputs = self._rnn(data.input)

        with tf.name_scope('Attention'):
            attention_output = self._attention(lstm_outputs)

        with tf.variable_scope('Affine', initializer=self.init):
            weights = tf.get_variable('weights',
                                        [lstm_units, output_units])
            biases = tf.get_variable('biases',
                                        [output_units])

        with tf.name_scope('Pred'):
            pred = tf.add(tf.matmul(attention_output, weights),
                          biases,
                          name='pred')
        self.pred = nn.softmax(pred)
        self.pred_label = tf.argmax(pred, 1)

        if self.mode != 'test':
            with tf.name_scope('Label'):
                label = tf.identity(data.label, 'label')

            with tf.name_scope('Loss'):
                loss = tf.add(
                    tf.reduce_mean(
                        nn.softmax_cross_entropy_with_logits(logits=pred,
                                                                labels=label)),
                    self.beta * tf.reduce_sum(nn.l2_loss(weights)),
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

        if self.mode == 'train':
            with tf.name_scope('TrainOp'):
                self.train_op = self.opt.minimize(self.loss)

    def _rnn(self, x):
        word_idx = x
        word_idx = tf.cast(word_idx, tf.int32)
        init = tf.constant(WordVec.word_vecs)
        with tf.variable_scope('Embeddings'):
            embeddings = tf.get_variable('embeddings', initializer=init)
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
                                        [self.lstm_units, self.output_units])
            biases = tf.get_variable('biases',
                                        [1, self.output_units])
            u_w = tf.get_variable('u_w',
                                        [self.output_units, 1])

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
            # v: (64, 1) -> (64,1)
            v = tf.reshape(v, [v.get_shape()[0], 1])
            # v: (64,1) -> [(64,1), (64,1), ...]
            v = [v] * self.lstm_units
            # v: (64,self.lstm_units)
            v = tf.concat(v, axis=1)
            outputs.append(tf.multiply(v, lstm_outputs[i]))

        return tf.add_n(outputs)


class WordVec(object):
    _word_weights_path = join(dirname(dirname(dirname(abspath(__file__)))),
                              'data',
                              'word_vec',
                              'word2vec.dat')
    with open(_word_weights_path, 'r') as f:
        word_vecs = pickle.load(f).astype(np.float32)
