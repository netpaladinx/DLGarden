from __future__ import absolute_import
from __future__ import  division
from __future__ import  print_function

import sys

import tensorflow as tf
import numpy as np

from vae import VAE, default_hparams
from train import train


def main(args):
    hparams = default_hparams
    hparams.parse(FLAGS.hparams)
    print(hparams.values())

    train(VAE, hparams, FLAGS)

if __name__ == '__main__':
    tf.flags.DEFINE_string("hparams", "", """Comma separated list of name=value pairs.""")
    tf.flags.DEFINE_boolean("debug", True, """Enabling debug for producing consistent results.""")
    tf.flags.DEFINE_string("data_dir", "./data", """Store downloaded data""")
    tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint_tmp", "")
    tf.flags.DEFINE_string("summary_dir", "./summary_tmp", "")
    tf.flags.DEFINE_integer("save_checkpoint_steps", 1000, "")
    tf.flags.DEFINE_integer("save_summaries_steps", 100, "")
    tf.flags.DEFINE_integer("display_steps", 100, "")
    tf.flags.DEFINE_integer("eval_steps", 1000, "")

    FLAGS = tf.flags.FLAGS
    if FLAGS.debug:
        np.random.seed(0)
        tf.set_random_seed(0)

    tf.app.run()
