"""
1. Use a Bernoulli distribution rather than a Gaussian distribution in the
   generator network
2. Use a scikit-learn-like interface: `partial_fit`
3. Train end-to-end
4. The model can be used (1) to reconstruct unseen input, (2) to generate new
   samples, and (3) to map inputs to the latent space

Paper: "Auto-Encoding Variational Bayes" (Kingma & Welling)
"""

from __future__ import absolute_import
from __future__ import  division
from __future__ import  print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers


class VAE(object):

    def __init__(self, hparams, train_dataset, test_dataset):
        self.hparams = hparams

        # Create the input
        self.x = self._create_input(train_dataset, test_dataset)
        self.batch_size = tf.shape(self.x)[0]

        # Create the network
        self._create_network()

        # Create the loss
        self._create_loss()

        # Create the optimizer
        self.global_step = tf.train.create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.init_op = tf.global_variables_initializer()

        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

        self.sess = None

    def _create_input(self, train_dataset, test_dataset):
        train_dataset = train_dataset.repeat(count=self.hparams.n_epochs).shuffle(self.hparams.shuffle_buffer).batch(self.hparams.batch_size)
        test_dataset = test_dataset.batch(self.hparams.batch_size)

        self.train_iterator = train_dataset.make_one_shot_iterator()
        self.test_iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
        self.test_init_op = self.test_iterator.make_initializer(test_dataset)
        self.train_handle = None
        self.test_handle = None

        self.dataset_handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
        iterator = tf.data.Iterator.from_string_handle(self.dataset_handle,
                                                       train_dataset.output_types,
                                                       train_dataset.output_shapes)
        next_element = iterator.get_next()
        x, _ = next_element
        return x

    def _create_network(self):
        with tf.variable_scope("VAE"):
            # Compute the mean and the log of variance of Gaussian distribution
            # in the latent space
            self.z_mean, self.z_log_variance = self._recognition_network(self.x)

            # Draw a sample from Gaussian distribution give mean and variance
            self.z = self._sampler(self.z_mean, self.z_log_variance)

            # Compute the mean of Bernoulli distribution of reconstructed input
            self.x_reconstr_logits, self.x_reconstr_mean = self._generator_network(self.z)

    def _recognition_network(self, x):
        """ Probabilistic encoder"""
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer()):
            h1 = slim.fully_connected(x, self.hparams.n_hidden_recog[0],
                                      scope='recog-1')
            h2 = slim.fully_connected(h1, self.hparams.n_hidden_recog[1],
                                      scope='recog-2')
            z_mean = slim.fully_connected(h2, self.hparams.n_latent,
                                          activation_fn=None,
                                          scope='latent-mean')
            z_log_variance = slim.fully_connected(h2, self.hparams.n_latent,
                                                  activation_fn=None,
                                                  scope='latent-variance')
            return z_mean, z_log_variance

    def _sampler(self, mean, log_variance):
        eps = tf.random_normal([self.batch_size, self.hparams.n_latent])
        z = tf.sqrt(tf.exp(log_variance)) * eps + mean
        return z

    def _generator_network(self, z):
        """ Probabilistic decoder
        """
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer()):
            h1 = slim.fully_connected(z, self.hparams.n_hidden_gener[0],
                                      scope='gener-1')
            h2 = slim.fully_connected(h1, self.hparams.n_hidden_gener[1],
                                      scope='gener-2')
            x_reconstr_logits = slim.fully_connected(h2,
                                                     self.hparams.n_input,
                                                     activation_fn=None,
                                                     scope='reconstr-mean')
            x_reconstr_mean = tf.sigmoid(x_reconstr_logits)
            return x_reconstr_logits, x_reconstr_mean

    def _create_loss(self):
        """loss = reconstruction_loss + Kullback-Leibler_divergence

        reconstruction_loss = - sum_i ( x_i * log (p_i + 1e-10) +
                                        (1-x_i) * log (1-p_i + 1e-10) )
        KL(N(mu1,sigma1), N(mu2,sigma2)) = log(sigma2) - log(sigma1) +
                                (sigma1^2 + (mu1 - mu2)^2) / 2 sigma2^2 - 1/2
        KL(N(mu,sigma), N(0,1)) = - log(sigma) + (sigma^2 + mu^2) / 2 - 1/2
        """
        reconstr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.x,
            logits=self.x_reconstr_logits,
            name='reconstruction_loss'),
            1)
        kl_loss = tf.reduce_sum((- self.z_log_variance + tf.exp(self.z_log_variance)
                                 + tf.square(self.z_mean) - 1) / 2,
                                1)
        # Average over batch
        self.loss = tf.reduce_mean(reconstr_loss + kl_loss)

    # Main Interfaces #

    def train_init(self, sess):
        self.train_handle = sess.run(self.train_iterator.string_handle())

    def eval_init(self, sess):
        self.test_handle = sess.run(self.test_iterator.string_handle())
        sess.run(self.test_init_op)

    def partial_fit(self, sess, x=None):
        """ Perform one SGD step and compute the cost

        x: mini-batch of input data
        """
        if x is None:
            _, loss = sess.run([self.train_op, self.loss],
                               feed_dict={self.dataset_handle: self.train_handle})
        else:
            _, loss = sess.run([self.train_op, self.loss],
                                feed_dict={self.x: x})
        return loss

    def partial_eval(self, sess, x=None):
        """ Compute the cost
        """
        if x is None:
            batch_size, loss = sess.run([self.batch_size, self.loss],
                                        feed_dict={self.dataset_handle: self.test_handle})
        else:
            loss = sess.run(self.loss, feed_dict={self.x: x})
            batch_size = 1
        return batch_size, loss

    def transform(self, sess, x):
        """ Compute the latent code given an input
        """
        z = sess.run(self.z_mean, feed_dict={self.x: x})
        return z

    def generate(self, sess, z_mean=None):
        """ Generate data from a drawn sample of latent code
        """
        if z_mean is None:
            z_mean = np.random.normal(size=self.hparams.n_latent)
        x_reconstr_mean = sess.run(self.x_reconstr_mean, feed_dict={self.z, z_mean})
        return x_reconstr_mean

    def reconstruct(self, sess, x):
        """ Reconstruct a given input
        """
        x_reconstr_mean = sess.run(self.x_reconstr_mean, feed_dict={self.x, x})
        return x_reconstr_mean

# for MNIST data
default_hparams = tf.contrib.training.HParams(
    n_input = 784,
    n_hidden_recog = [512, 512],
    n_hidden_gener = [512, 512],
    n_latent = 20,
    learning_rate = 0.001,
    batch_size = 128,
    shuffle_buffer = 1000,
    n_epochs = 10
)
