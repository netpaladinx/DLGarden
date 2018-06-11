from __future__ import absolute_import
from __future__ import  division
from __future__ import  print_function

import os

import tensorflow as tf

from dataset import train_dataset, test_dataset
from utils import SessionWithouHooks


def train(model_cls, hparams, FLAGS):
    model = model_cls(hparams, train_dataset(FLAGS.data_dir), test_dataset(FLAGS.data_dir))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir = FLAGS.checkpoint_dir,
        save_checkpoint_steps = FLAGS.save_checkpoint_steps,
        hooks = [tf.train.StopAtStepHook(last_step=2001),
                 tf.train.SummarySaverHook(save_steps=FLAGS.save_summaries_steps,
                                           output_dir=FLAGS.summary_dir,
                                           summary_op=model.summary_op)],
        config=tf.ConfigProto(allow_soft_placement=True,
                              gpu_options=tf.GPUOptions(allow_growth=True,
                                                        visible_device_list='0'))  # Only use "/device:GPU:0"
    ) as sess:
        sess_no_hooks = SessionWithouHooks(sess)
        model.train_init(sess_no_hooks)

        step = 1
        while not sess.should_stop():
            loss = model.partial_fit(sess)

            if step % FLAGS.display_steps == 0:
                print("Step: %d | Loss: %.5f" % (step, loss))

            if step % FLAGS.eval_steps == 0:
                eval(sess_no_hooks, model)

            step += 1

def eval(sess, model):
    try:
        model.eval_init(sess)

        avg_loss = 0.
        n_samples = 0

        while True:
            batch_size, loss = model.partial_eval(sess)

            alpha = n_samples * 1. / (n_samples + batch_size)
            avg_loss = avg_loss * alpha + loss * (1-alpha)
            n_samples += batch_size
    except tf.errors.OutOfRangeError:
        print('[Eval] Avg Loss: %.5f' % avg_loss)