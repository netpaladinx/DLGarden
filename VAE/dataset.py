"""
Download four data files from http://yann.lecun.com/exdb/mnist/:
    train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
    train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
    t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
    t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

tf.data.Dataset interface to the MNIST dataset
"""

from __future__ import absolute_import
from __future__ import  division
from __future__ import  print_function

import os
import shutil
import tempfile
import gzip

import numpy as np
from six.moves import urllib
import tensorflow as tf

data_url = "http://yann.lecun.com/exdb/mnist/"


def _read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer"""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset"""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = _read32(f)
        _read32(f)  # num_images, unused
        rows = _read32(f)
        cols = _read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic, f.name))
        if rows != 28 or cols != 28:
            raise ValueError('Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                             (f.name, rows, cols))


def _check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset"""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = _read32(f)
        _read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic, f.name))


def _download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done"""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    url = data_url + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
        tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def _dataset(directory, images_file, labels_files):
    """Download and parse MNIST datasets."""
    images_file = _download(directory, images_file)
    labels_files = _download(directory, labels_files)

    _check_image_file_header(images_file)
    _check_labels_file_header(labels_files)

    # Tutorial: show how to use tf.data.Dataset

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.unit8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_files, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


# Dataset API #


def train_dataset(directory):
    """tf.data.Dataset object for MNIST training data

    `directory`: the directory to store `train-images-idx3-ubyte` and `train-labels-idx1-ubyte`
    """
    return _dataset(directory, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')


def test_dataset(directory):
    """tf.data.Dataset object for MNIST test data

    `directory`: the directory to store `t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte`
    """
    return _dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')