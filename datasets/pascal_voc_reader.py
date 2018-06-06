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
TRAIN_DIR = '/mnt/disk50_CHENYIFENG/VOC2012/tf_multitask/train'
VALIDATION_DIR = '/mnt/disk50_CHENYIFENG/VOC2012/tf_multitask/val'
TRAIN_NUM = 10582
VALID_NUM = 1449


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
            'label/segmentation/format': tf.FixedLenFeature([], tf.string),
            'label/segmentation/encoded': tf.FixedLenFeature([], tf.string),
            'label/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'label/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'label/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'label/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'label/object/bbox/label': tf.VarLenFeature(tf.int64),
            'label/object/bbox/difficult': tf.VarLenFeature(tf.int64),
            'label/object/bbox/truncated': tf.VarLenFeature(tf.int64)
        })

    features['image/encoded'] = image_ops.decode_jpeg(features['image/encoded'], channels=3)
    features['label/segmentation/encoded'] = image_ops.decode_png(features['label/segmentation/encoded'], channels=1)

    return features


def extract(features):
    image = features['image/encoded']
    segmentation = features['label/segmentation/encoded']
    name = features['image/name']

    # read data in the format xmins, ymins, xmaxs, ymaxes as writer writes
    xmins = tf.sparse_tensor_to_dense(features['label/object/bbox/xmin'])
    ymins = tf.sparse_tensor_to_dense(features['label/object/bbox/ymin'])
    xmaxs = tf.sparse_tensor_to_dense(features['label/object/bbox/xmax'])
    ymaxs = tf.sparse_tensor_to_dense(features['label/object/bbox/ymax'])
    bboxes_labels = tf.sparse_tensor_to_dense(features['label/object/bbox/label'])
    # stack use standard order
    bboxes = tf.transpose(tf.stack([ymins, xmins, ymaxs, xmaxs]))

    return name, image, segmentation, bboxes, bboxes_labels


def augment(name, image, segmentation, bboxes, bboxes_labels):
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    return name, image, segmentation, bboxes, bboxes_labels


def normalize(name, image, segmentation, bboxes, bboxes_labels):
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return name, image, segmentation, bboxes, bboxes_labels


def reshape(name, image, segmentation, bboxes, bboxes_labels, reshape_size=None):
    if reshape_size is not None:
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bilinear(image, reshape_size)
        segmentation = tf.expand_dims(segmentation, axis=0)
        segmentation = tf.image.resize_nearest_neighbor(segmentation, reshape_size)

    return name, tf.squeeze(image, axis=0), tf.squeeze(segmentation, axis=0), bboxes, bboxes_labels


def cast_type(name, image, segmentation, bboxes, bboxes_labels):
    return name, tf.cast(image, tf.float32), tf.cast(segmentation, tf.int32), bboxes, tf.cast(bboxes_labels, tf.int32)


def get_dataset(dir, batch_size, num_epochs, reshape_size, padding='SAME', normalize=True):
    """Reads input data num_epochs times. AND Return the dataset

    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
      padding:  if 'SAME' , have ceil(#samples / batch_size) * epoch_nums batches
                if 'VALID', have floor(#samples / batch_size) * epoch_nums batches
      normalize: whether normalize

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
        if normalize:
            dataset = dataset.map(normalize)
        dataset = dataset.map(set_parameter(reshape, reshape_size=reshape_size))

        # the parameter is the queue size
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.batch(batch_size)
    return dataset


def get_next_batch(dataset):
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
