#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import sys
from os.path import abspath, dirname, join

import tensorflow as tf

sys.path.append(dirname(dirname(abspath(__file__))))
from data.decode_tfrecords import read_and_decode
from model.lstm_model import LSTM_attention
from util import get_num_records, get_cfg_path

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.INFO)

# Model Parameters
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to run trainer.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('keep_prob', 0.7, 'The keep probability for lstm dropout.')
flags.DEFINE_float('beta', 0.0001, 'The regularization term for l2 norm.')
flags.DEFINE_integer('lstm_units', 100, 'lstm output units.')
flags.DEFINE_integer('output_units', 5, 'output units.')
flags.DEFINE_integer('early_stop_epochs',
                     3,
                     'The maximum number of times when validation gets worse.')

# Other Parameters
flags.DEFINE_integer('gpu', 0, 'Which gpu to use.')
flags.DEFINE_string('save_path',
                    '/tmp/model',
                    'Model output directory.')

FLAGS = flags.FLAGS

cf = ConfigParser.ConfigParser()
cf.read(get_cfg_path())

CHECKPOINT_BASENAME = 'model.ckpt'


class Input(object):
    """ The input data. """

    def __init__(self, mode):
        input, label, records_num = self._get_data(name)

        self.input = input
        self.label = label
        self.records_num = records_num

    @staticmethod
    def _get_data(mode):
        filename = join(dirname(dirname(dirname(abspath(__file__)))),
                        'data',
                        'tfrecords_data',
                        mode + '.tfrecords')

        num_epochs = None
        records_num = get_num_records(filename)
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=num_epochs)
        if mode == 'test':
            input, _, _ = read_and_decode(filename_queue, mode)
            batch_size = records_num
            input = tf.train.batch(input,
                                   batch_size=batch_size,
                                   num_threads=2,
                                   capacity=1000 + 3 * batch_size)
            return input, records_num
        elif mode == 'dev':
            input, label, _, _ = read_and_decode(filename_queue, mode)
            label = tf.one_hot(label, FLAGS.output_units, name='label')
            batch_size = records_num
            input, label = tf.train.batch([input label],
                                   batch_size=batch_size,
                                   num_threads=2,
                                   capacity=1000 + 3 * batch_size)
            return input, label, records_num
        elif mode == 'train':
            input, label, _, _ = read_and_decode(filename_queue, mode)
            label = tf.one_hot(label, FLAGS.output_units, name='label')
            batch_size = records_num
            input, label = tf.train.shuffle_batch([input label],
                                   batch_size=batch_size,
                                   num_threads=2,
                                   capacity=1000 + 3 * batch_size,
                                   min_after_dequeue=1000)
            return input, label, records_num


def _run_training():
    init = tf.random_uniform_initializer(-0.1, 0.1)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    with tf.device('/gpu:%d' % FLAGS.gpu):
        with tf.name_scope('Train'):
            with tf.name_scope('TrainInput'):
                train_data = Input('train')
            with tf.variable_scope('Model', reuse=None):
                train_model = LSTM_attention(data = train_data,
                                             mode='train',
                                             lstm_units=FLAGS.lstm_units,
                                             output_units=FLAGS.output_units,
                                             opt = opt,
                                             init = init,
                                             beta=FLAGS.beta,
                                             keep_prob=FLAGS.keep_prob)

        with tf.name_scope('Dev'):
            with tf.name_scope('DevInput'):
                dev_data = Input('dev')
            with tf.variable_scope('Model', reuse=True):
                dev_model = LSTM_attention(data = dev_data,
                                           mode='dev',
                                           lstm_units=FLAGS.lstm_units,
                                           output_units=FLAGS.output_units,
                                           opt = opt,
                                           init = init,
                                           beta=FLAGS.beta,
                                           keep_prob=FLAGS.keep_prob)

        with tf.name_scope('Test'):
            with tf.name_scope('TestInput'):
                test_data = Input('test')
            with tf.variable_scope('Model', reuse=True):
                test_model = LSTM_attention(data = test_data,
                                            mode='test',
                                            lstm_units=FLAGS.lstm_units,
                                            output_units=FLAGS.output_units,
                                            opt = opt,
                                            init = init,
                                            beta=FLAGS.beta,
                                            keep_prob=FLAGS.keep_prob)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=FLAGS.save_path,
                             save_model_secs=600,
                             save_summaries_secs=120)

    with sv.managed_session(config=sess_config) as sess:
        for epoch in xrange(FLAGS.num_epochs):
            if sv.should_stop():
                break
            step_num = train_model.data.records_num // FLAGS.batch_size - 1
            for step in xrange(step_num):
                _, train_loss, train_acc = sess.run(
                    [train_model.train_op,
                     train_model.loss,
                     train_model.accuracy])
                if epoch == 0 and step == 0:
                    avg_loss, avg_acc = train_loss, train_acc
                else:
                    avg_loss = 0.9 * avg_loss + 0.1 * train_loss
                    avg_acc = 0.9 * avg_acc + 0.1 * train_acc
                logging.info('training loss for epoch %d step %d is %f'
                             % (epoch + 1, step + 1, avg_loss))

            dev_loss, dev_acc, dev_l2 = sess.run([dev_model.loss,
                                                  dev_model.accuracy])
            logging.info(
                'train loss for epoch %d is %f, accuracy is %f'
                % (epoch + 1, avg_loss, avg_acc))
            logging.info(
                'dev   loss for epoch %d is %f, accuracy is %f'
                % (epoch + 1, dev_loss, dev_acc))

        sv.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.save_path))
        test_pred_label = sess.run(test_model.pred_label)


def main(_):
    _run_training()


if __name__ == '__main__':
    tf.app.run()