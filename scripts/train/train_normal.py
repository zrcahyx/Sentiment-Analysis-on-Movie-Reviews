#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import sys
from os.path import abspath, dirname, join

import tensorflow as tf

sys.path.append(dirname(dirname(abspath(__file__))))
from data.decode_tfrecords import read_and_decode
from model.my_model import My_model, Input
from util import get_num_records, get_cfg_path

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.INFO)

# Model Parameters
flags.DEFINE_integer('num_epochs', 20, 'Number of epochs to run trainer.')
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_float('keep_prob', 0.4, 'The keep probability for lstm dropout.')
flags.DEFINE_float('beta', 0.01, 'The regularization term for l2 norm.')
flags.DEFINE_integer('lstm_units', 100, 'lstm output units.')
flags.DEFINE_integer('cnn_flag', 1, 'Use cnn feature or not.')
flags.DEFINE_integer('cnn_kernels', 20, 'How many filters to use for CNN.')
flags.DEFINE_string('cnn_ngrams', '2,3,4,5,6', 'cnn filter size.')
flags.DEFINE_integer('rnn_flag', 1, 'Use rnn feature or not.')
flags.DEFINE_string('hidden_units', '0', 'hidden units.')
flags.DEFINE_integer('output_units', 5, 'output units.')

# Other Parameters
flags.DEFINE_integer('gpu', 0, 'Which gpu to use.')
flags.DEFINE_string('save_path',
                    '/tmp/model',
                    'Model output directory.')

FLAGS = flags.FLAGS

cf = ConfigParser.ConfigParser()
cf.read(get_cfg_path())
cf.set('Model', 'learning_rate', FLAGS.learning_rate)
cf.set('Model', 'batch_size', FLAGS.batch_size)
cf.set('Model', 'lstm_units', FLAGS.lstm_units)
cf.set('Model', 'cnn_flag', FLAGS.cnn_flag)
cf.set('Model', 'cnn_kernels', FLAGS.cnn_kernels)
cf.set('Model', 'cnn_ngrams', FLAGS.cnn_ngrams)
cf.set('Model', 'rnn_flag', FLAGS.rnn_flag)
cf.set('Model', 'hidden_units', FLAGS.hidden_units)
cf.set('Model', 'output_units', FLAGS.output_units)
cf.set('Model', 'beta', FLAGS.beta)
cf.set('Model', 'keep_prob', FLAGS.keep_prob)
cf.set('Model', 'save_path', FLAGS.save_path)
with open(get_cfg_path(), 'w') as f:
        cf.write(f)

if FLAGS.hidden_units == '0':
    hidden_units = []
else:
    hidden_units=[int(x) for x in FLAGS.hidden_units.split(',')]

cnn_ngrams = [int(x) for x in FLAGS.cnn_ngrams.split(',')]


CHECKPOINT_BASENAME = 'model.ckpt'


def _run_training():
    lr = FLAGS.learning_rate
    # opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    with tf.device('/gpu:%d' % FLAGS.gpu):
        with tf.name_scope('Train'):
            with tf.name_scope('TrainInput'):
                train_data = Input('train')
            with tf.variable_scope('Model', reuse=None):
                train_model = My_model(data = train_data,
                                       mode='train',
                                       lstm_units=FLAGS.lstm_units,
                                       cnn_flag=FLAGS.cnn_flag,
                                       cnn_kernels=FLAGS.cnn_kernels,
                                       cnn_ngrams=cnn_ngrams,
                                       rnn_flag=FLAGS.rnn_flag,
                                       hidden_units=hidden_units,
                                       output_units=FLAGS.output_units,
                                       beta=FLAGS.beta,
                                       keep_prob=FLAGS.keep_prob)

        with tf.name_scope('Dev'):
            with tf.name_scope('DevInput'):
                dev_data = Input('dev')
            with tf.variable_scope('Model', reuse=True):
                dev_model = My_model(data = dev_data,
                                     mode='dev',
                                     lstm_units=FLAGS.lstm_units,
                                     cnn_flag=FLAGS.cnn_flag,
                                     cnn_kernels=FLAGS.cnn_kernels,
                                     cnn_ngrams=cnn_ngrams,
                                     rnn_flag=FLAGS.rnn_flag,
                                     hidden_units=hidden_units,
                                     output_units=FLAGS.output_units,
                                     beta=FLAGS.beta,
                                     keep_prob=FLAGS.keep_prob)

    with tf.name_scope('TrainOp'):
        train_op = opt.minimize(train_model.loss)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=FLAGS.save_path,
                             save_model_secs=600,
                             save_summaries_secs=120)

    with sv.managed_session(config=sess_config) as sess:
        # dev_acc_past = 0
        for epoch in xrange(FLAGS.num_epochs):
            if sv.should_stop():
                break
            step_num = train_model.data.records_num // FLAGS.batch_size - 1
            for step in xrange(step_num):
                _, train_loss, train_acc = sess.run(
                    [train_op,
                     train_model.loss,
                     train_model.accuracy])
                if epoch == 0 and step == 0:
                    avg_loss, avg_acc = train_loss, train_acc
                else:
                    avg_loss = 0.9 * avg_loss + 0.1 * train_loss
                    avg_acc = 0.9 * avg_acc + 0.1 * train_acc
                logging.info('training loss for epoch %d step %d is %f'
                             % (epoch + 1, step + 1, avg_loss))

            dev_loss, dev_acc = sess.run([dev_model.loss,
                                                  dev_model.accuracy])
            # if dev_acc < dev_acc_past:
            #     lr /= 2
            #     opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
            #     with tf.name_scope('TrainOp'):
            #         train_op = opt.minimize(train_model.loss)
            # dev_acc_past = dev_acc

            logging.info(
                'train loss for epoch %d is %f, accuracy is %f'
                % (epoch + 1, avg_loss, avg_acc))
            logging.info(
                'dev   loss for epoch %d is %f, accuracy is %f'
                % (epoch + 1, dev_loss, dev_acc))


def main(_):
    _run_training()


if __name__ == '__main__':
    tf.app.run()
