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
TRAIN_DIR = '/mnt/disk50/datasets/COCO/tf_records/detection/train2017'
VALIDATION_DIR = '/mnt/disk50/datasets/COCO/tf_records/detection/val2017'


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/name': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'label/num_classes': tf.FixedLenFeature([], tf.int64),
            'label/masks': tf.FixedLenFeature([], tf.string),
            'label/bboxes': tf.FixedLenFeature([], tf.string),
            'label/classes': tf.FixedLenFeature([], tf.string)
        }
    )

    features['image/encoded'] = image_ops.decode_jpeg(features['image/encoded'], channels=3)
    features['label/masks'] = tf.decode_raw(features['label/masks'], tf.float64)
    features['label/bboxes'] = tf.decode_raw(features['label/bboxes'], tf.float64)
    features['label/classes'] = tf.decode_raw(features['label/classes'], tf.float64)

    ih = tf.cast(features['image/height'], tf.int32)
    iw = tf.cast(features['image/width'], tf.int32)
    num_classes = tf.cast(features['label/num_classes'], tf.int32)

    features['label/masks'] = tf.cast(tf.reshape(features['label/masks'], [ih, iw, num_classes]), tf.int64)
    features['label/bboxes'] = tf.cast(tf.reshape(features['label/bboxes'], [num_classes, 4]), tf.float64)
    features['label/classes'] = tf.cast(tf.reshape(features['label/classes'], [num_classes, 1]), tf.int64)

    return features


def extract(features):
    name = features['image/name']
    image = features['image/encoded']
    masks = features['label/masks']
    bboxes = features['label/bboxes']
    classes = features['label/classes']
    num_classes = features['label/num_classes']
    label = {'masks': masks, 'bboxes': bboxes, 'classes': classes, 'num_classes': num_classes}
    return name, image, label


def augment(name, image, label):
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    return name, image, label


def normalize(name, image, label):
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return name, image, label


def reshape(name, image, label, reshape_size=None):
    if reshape_size is not None:
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bilinear(image, reshape_size)
        masks = label['masks']
        masks = tf.expand_dims(masks, axis=0)
        num_classes = label['num_classes']
        masks = tf.image.resize_nearest_neighbor(masks, reshape_size + [num_classes])
        masks = tf.squeeze(masks, axis=0)
        label['masks'] = masks

    return name, tf.squeeze(image, axis=0), label


def cast_type(name, image, label):
    image = tf.cast(image, tf.float32)
    label['masks'] = tf.cast(label['masks'], tf.int32)
    label['bboxes'] = tf.cast(label['bboxes'], tf.float32)
    label['classes'] = tf.cast(label['classes'], tf.int32)
    label['num_classes'] = tf.cast(label['num_classes'], tf.int32)

    return name, image, label



def COCO_get_datasets(dir, batch_size, num_epochs, reshape_size):
    """Reads input data num_epochs times. AND Return the dataset

    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
      padding:  if 'SAME' , have ceil(#samples / batch_size) * epoch_nums batches
                if 'VALID', have floor(#samples / batch_size) * epoch_nums batches

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
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.batch(batch_size)
    return dataset


def get_next_batch(dataset):
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()