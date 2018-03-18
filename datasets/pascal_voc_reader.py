# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train and Eval the MNIST network.

This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files
for context.

YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from functools import partial as set_parameter

import tensorflow as tf
from tensorflow.python.ops import image_ops

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_DIR = '/mnt/disk/chenyifeng/VOC2012/tf_records/tf_records/train'
VALIDATION_DIR = '/mnt/disk/chenyifeng/VOC2012/tf_records/tf_records/val'


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/name': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'label/format': tf.FixedLenFeature([], tf.string),
            'label/encoded': tf.FixedLenFeature([], tf.string)
        })

    features['image/encoded'] = image_ops.decode_jpeg(features['image/encoded'], channels=3)
    features['label/encoded'] = image_ops.decode_png(features['label/encoded'], channels=1)
    # image.set_shape((mnist.IMAGE_PIXELS))
    return features


def extract(features):
    image = features['image/encoded']
    label = features['label/encoded']
    name = features['image/name']
    return name, image, label


def augment(name, image, label):
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    return name, image, label


def normalize(name, image, label):
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return name, image, label


def reshape(name, image, label, reshape_size=None):
    if reshape_size is not None:
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bilinear(image, reshape_size)
        label = tf.expand_dims(label, axis=0)
        label = tf.image.resize_nearest_neighbor(label, reshape_size)

    return name, tf.squeeze(image), tf.squeeze(label)


def cast_type(name, image, label):
    return name, tf.cast(image, tf.float32), tf.cast(label, tf.int32)


def pascal_inputs(dir, batch_size, num_epochs, reshape_size):
    """Reads input data num_epochs times.

    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.

    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).

      This function creates a one_shot_iterator, meaning that it will only iterate
      over the dataset once. On the other hand there is no special initialization
      required.
    """
    if not num_epochs:
        num_epochs = None
    filenames = [os.path.join(dir, i) for i in os.listdir(dir)]

    with tf.name_scope('input'):
        # TFRecordDataset opens a protobuf and reads entries line by line
        # could also be [list, of, filenames]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat(num_epochs)

        # map takes a python function and applies it to every sample
        dataset = dataset.map(decode)
        dataset = dataset.map(extract)
        dataset = dataset.map(cast_type)
        dataset = dataset.map(augment)
        dataset = dataset.map(normalize)
        dataset = dataset.map(set_parameter(reshape, reshape_size=reshape_size))

        # the parameter is the queue size
        # dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

        name_batch, image_batch, label_batch = iterator.get_next()
        image_batch = tf.reshape(image_batch, [batch_size] + reshape_size + [3])
        label_batch = tf.reshape(label_batch, [batch_size] + reshape_size + [1])

    return name_batch, image_batch, label_batch
