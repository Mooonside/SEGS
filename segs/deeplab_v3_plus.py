# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
r"""Provides DeepLab model definition and helper functions.

DeepLab is a deep learning system for semantic image segmentation with
the following features:

(1) Atrous convolution to explicitly control the resolution at which
feature responses are computed within Deep Convolutional Neural Networks.

(2) Atrous spatial pyramid pooling (ASPP) to robustly segment objects at
multiple scales with filters at multiple sampling rates and effective
fields-of-views.

(3) ASPP module augmented with image-level feature and batch normalization.

(4) A simple yet effective decoder module to recover the object boundaries.

See the following papers for more details:

"Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation"
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
(https://arxiv.org/abs/1802.02611)

"Rethinking Atrous Convolution for Semantic Image Segmentation,"
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
(https://arxiv.org/abs/1706.05587)

"DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
Atrous Convolution, and Fully Connected CRFs",
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L Yuille (* equal contribution)
(https://arxiv.org/abs/1606.00915)

"Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected
CRFs"
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L. Yuille (* equal contribution)
(https://arxiv.org/abs/1412.7062)
"""
import numpy as np
import tensorflow as tf

from backbones import feature_extractor
from segs.common_configure import DeepLabFlags
from tf_ops.wrap_ops import conv2d, sep_conv2d, drop_out, arg_scope, regularizer, \
    batch_norm2d, avg_pool2d, ms_softmax_with_logits

_LOGITS_SCOPE_NAME = 'logits'
_MERGED_LOGITS_SCOPE = 'merged_logits'
_IMAGE_POOLING_SCOPE = 'image_pooling'
_ASPP_SCOPE = 'aspp'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_DECODER_SCOPE = 'decoder'
_DEBUG_SCOPE = 'debug'


class DEBUG:
    def __init__(self):
        pass


DEBUG_VARS = DEBUG()


def get_extra_layer_scopes():
    """Gets the scopes for extra layers.

    Returns:
      A list of scopes for extra layers.
    """
    return [
        _LOGITS_SCOPE_NAME,
        _IMAGE_POOLING_SCOPE,
        _ASPP_SCOPE,
        _CONCAT_PROJECTION_SCOPE,
        _DECODER_SCOPE,
    ]


def predict_labels(images, model_options, image_pyramid=None):
    """Predicts segmentation labels.

    Args:
      images: A tensor of size [batch, height, width, channels].
      model_options: A ModelOptions instance to configure models.
      image_pyramid: Input image scales for multi-scale feature extraction.

    Returns:
      A dictionary with keys specifying the output_type (e.g., semantic
        prediction) and values storing Tensors representing predictions (argmax
        over channels). Each prediction has size [batch, height, width].
    """
    outputs_to_scales_to_logits = multi_scale_logits(
        images,
        model_options=model_options,
        image_pyramid=image_pyramid,
        is_training=False,
        fine_tune_batch_norm=False)

    predictions = {}
    for output in sorted(outputs_to_scales_to_logits):
        scales_to_logits = outputs_to_scales_to_logits[output]
        logits = tf.image.resize_bilinear(
            scales_to_logits[_MERGED_LOGITS_SCOPE],
            tf.shape(images)[1:3],
            align_corners=True)
        predictions[output] = tf.argmax(logits, 3)

    return predictions


def scale_dimension(dim, scale):
    """Scales the input dimension.

    Args:
      dim: Input dimension (a scalar or a scalar Tensor).
      scale: The amount of scaling applied to the input.

    Returns:
      Scaled dimension.

    TODO: cast_int = floor(), floor((y - 1) / x + 1) = ceil(y / x)
    """
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
    else:
        return int((float(dim) - 1.0) * scale + 1.0)


def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):
    """Gets the logits for multi-scale inputs.

    The returned logits are all downsampled (due to max-pooling layers)
    for both training and evaluation.

    Args:
      images: A tensor of size [batch, height, width, channels].
      model_options: A ModelOptions instance to configure models.
      image_pyramid: Input image scales for multi-scale feature extraction.

      weight_decay: The weight decay for model variables.
      is_training: Is training or not.
      fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

    Returns:
        {
            'TASK_NAME':{
                Image_Scale : feature
                ......
                _MERGED_LOGITS_SCOPE : merged feature

            }
            ...(IF MORE TASKS)
        }

      outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
        semantic prediction) to a dictionary of multi-scale logits names to
        logits. For each output_type, the dictionary has keys which
        correspond to the scales and values which correspond to the logits.
        For example, if `scales` equals [1.0, 1.5], then the keys would
        include 'merged_logits', 'logits_1.00' and 'logits_1.50'.

    Raises:
      ValueError: If model_options doesn't specify crop_size and its
        add_image_level_feature = True, since add_image_level_feature requires
        crop_size information.
    """
    # Setup default values.
    if not image_pyramid:
        image_pyramid = [1.0]

    if model_options.crop_size is None and model_options.add_image_level_feature:
        raise ValueError(
            'Crop size must be specified for using image-level feature.')
    if model_options.model_variant == 'mobilenet_v2':
        if (model_options.atrous_rates is not None or
                model_options.decoder_output_stride is not None):
            # Output a warning and users should make sure if the setting is desired.
            tf.logging.warning('Our provided mobilenet_v2 checkpoint does not '
                               'include ASPP and decoder modules.')

    crop_height = (
        # 514
        model_options.crop_size[0]
        if model_options.crop_size else tf.shape(images)[1])
    crop_width = (
        model_options.crop_size[1]
        if model_options.crop_size else tf.shape(images)[2])

    # Compute the height, width for the output logits.
    # default to 16 , i.e. final predictions is [H/16, W/16]
    logits_output_stride = (
            model_options.decoder_output_stride or model_options.output_stride)

    logits_height = scale_dimension(
        crop_height,
        max(1.0, max(image_pyramid)) / logits_output_stride)
    logits_width = scale_dimension(
        crop_width,
        max(1.0, max(image_pyramid)) / logits_output_stride)

    # Compute the logits for each scale in the image pyramid.
    outputs_to_scales_to_logits = {
        k: {}
        for k in model_options.outputs_to_num_classes
    }

    for count, image_scale in enumerate(image_pyramid):
        if image_scale != 1.0:
            scaled_height = scale_dimension(crop_height, image_scale)
            scaled_width = scale_dimension(crop_width, image_scale)
            scaled_crop_size = [scaled_height, scaled_width]
            scaled_images = tf.image.resize_bilinear(
                images, scaled_crop_size, align_corners=True)
            if model_options.crop_size:
                scaled_images.set_shape([None, scaled_height, scaled_width, 3])
        else:
            scaled_crop_size = model_options.crop_size
            scaled_images = images

        model_options.crop_size = scaled_crop_size
        outputs_to_logits = _get_logits(
            scaled_images,
            model_options,
            weight_decay=weight_decay,
            reuse=tf.AUTO_REUSE,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)

        # Resize the logits to have the same dimension before merging.
        for output in sorted(outputs_to_logits):
            # resize_bilinear requires channel to be one or three
            outputs_to_logits[output] = tf.image.resize_bilinear(
                outputs_to_logits[output], [logits_height, logits_width],
                align_corners=True)

        # Return when only one input scale.
        if len(image_pyramid) == 1:
            for output in sorted(model_options.outputs_to_num_classes):
                outputs_to_scales_to_logits[output][
                    _MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
            return outputs_to_scales_to_logits

        # Save logits to the output map.
        for output in sorted(model_options.outputs_to_num_classes):
            outputs_to_scales_to_logits[output][
                'logits_%.2f' % image_scale] = outputs_to_logits[output]

    # Merge the logits from all the multi-scale inputs.
    for output in sorted(model_options.outputs_to_num_classes):
        # Concatenate the multi-scale logits for each output type.
        all_logits = [
            tf.expand_dims(logits, axis=4)
            for logits in outputs_to_scales_to_logits[output].values()
        ]
        all_logits = tf.concat(all_logits, 4)
        merge_fn = (
            tf.reduce_max
            if model_options.merge_method == 'max' else tf.reduce_mean)
        outputs_to_scales_to_logits[output][_MERGED_LOGITS_SCOPE] = merge_fn(
            all_logits, axis=4)

    return outputs_to_scales_to_logits


def _extract_features(images,
                      model_options,
                      weight_decay=0.0001,
                      reuse=tf.AUTO_REUSE,
                      is_training=False,
                      fine_tune_batch_norm=False):
    """Extracts features by the particular model_variant.

    Args:
      images: A tensor of size [batch, height, width, channels].
      model_options: A ModelOptions instance to configure models.
      weight_decay: The weight decay for model variables.
      reuse: Reuse the model variables or not.
      is_training: Is training or not.
      fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

    Returns:
      concat_logits: A tensor of size [batch, feature_height, feature_width,
        feature_channels], where feature_height/feature_width are determined by
        the images height/width and output_stride.
      end_points: A dictionary from components of the network to the corresponding
        activation.
    """
    # feature extractor is a backbone factory
    DEBUG_VARS.raw_image = images
    features, end_points = feature_extractor.extract_features(
        images,
        output_stride=model_options.output_stride,
        multi_grid=model_options.multi_grid,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    # TODO:check
    # DEBUG_VARS.xception_feature = end_points['xception_65/entry_flow/conv1_1/Relu:0']
    DEBUG_VARS.xception_feature = features
    if not model_options.aspp_with_batch_norm:
        return features, end_points
    else:
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'eps': 1e-5,
            'affine': True,
        }
        regularize_func = regularizer('l2', weight_decay)
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            with arg_scope([sep_conv2d], activate=tf.nn.relu, activate_middle=tf.nn.relu, batch_norm=True,
                           depthwise_weight_reg=None, pointwise_weight_reg=regularize_func,
                           padding='SAME', strides=[1, 1]):
                with arg_scope([conv2d], activate=tf.nn.relu, weight_reg=regularize_func,
                               batch_norm=True, padding='SAME', strides=[1, 1]):
                    # TODO: ASPP IS IMPLEMENTED HERE! Check Out!
                    with arg_scope([batch_norm2d], **batch_norm_params):
                        depth = 256
                        branch_logits = []

                        # TODO: ADD IMAGE POOLING HERE
                        if model_options.add_image_level_feature:
                            # this crop size has been updated to the new scaled one outside, which is the exact size
                            # of this model's inputs
                            pool_height = scale_dimension(model_options.crop_size[0],
                                                          1. / model_options.output_stride)
                            pool_width = scale_dimension(model_options.crop_size[1],
                                                         1. / model_options.output_stride)
                            # global average pooling, check whether the shape here is 1?
                            image_feature = avg_pool2d(
                                features, [pool_height, pool_width], [pool_height, pool_width],
                                padding='VALID')
                            # collapse channels to depth after GAP
                            image_feature = conv2d(
                                inputs=image_feature, outc=depth, ksize=[1, 1], name=_IMAGE_POOLING_SCOPE)
                            # TODO:check
                            DEBUG_VARS.image_feature = image_feature
                            # reshape it to final feature map shape
                            image_feature = tf.image.resize_bilinear(
                                image_feature, [pool_height, pool_width], align_corners=True)
                            image_feature.set_shape([None, pool_height, pool_width, depth])
                            # add image level feature to branch_logits
                            branch_logits.append(image_feature)

                        # Employ a 1x1 convolution.
                        branch_logits.append(conv2d(features, outc=depth, ksize=[1, 1], name=_ASPP_SCOPE + str(0)))

                        if model_options.atrous_rates:
                            # Employ 3x3 convolutions with different atrous rates.
                            DEBUG_VARS.aspp_features = []
                            for i, rate in enumerate(model_options.atrous_rates, 1):
                                scope = _ASPP_SCOPE + str(i)
                                if model_options.aspp_with_separable_conv:
                                    aspp_features = sep_conv2d(
                                        features, outc=depth, ksize=[3, 3], ratios=[rate, rate], name=scope)
                                    DEBUG_VARS.aspp_features.append(aspp_features)
                                else:
                                    aspp_features = conv2d(
                                        features, outc=depth, ksize=[3, 3], ratios=[rate, rate], name=scope)
                                branch_logits.append(aspp_features)

                        # Merge branch logits.
                        concat_logits = tf.concat(branch_logits, 3)
                        DEBUG_VARS.aspp_concat_feature = concat_logits
                        concat_logits = conv2d(inputs=concat_logits, outc=depth, ksize=[1, 1],
                                               name=_CONCAT_PROJECTION_SCOPE)
                        concat_logits = drop_out(concat_logits, kp_prob=0.9, is_training=is_training,
                                                 name=_CONCAT_PROJECTION_SCOPE + '_dropout')

                        return concat_logits, end_points


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=tf.AUTO_REUSE,
                is_training=False,
                fine_tune_batch_norm=False):
    """Gets the logits by atrous/image spatial pyramid pooling.

    Args:
      images: A tensor of size [batch, height, width, channels].
      model_options: A ModelOptions instance to configure models.
      weight_decay: The weight decay for model variables.
      reuse: Reuse the model variables or not.
      is_training: Is training or not.
      fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

    Returns:
      outputs_to_logits: A map from output_type to logits.
    """
    features, end_points = _extract_features(
        images,
        model_options,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)
    # TODO: CHECK
    DEBUG_VARS.aspp_result = features
    if model_options.decoder_output_stride is not None:
        decoder_height = scale_dimension(model_options.crop_size[0],
                                         1.0 / model_options.decoder_output_stride)
        decoder_width = scale_dimension(model_options.crop_size[1],
                                        1.0 / model_options.decoder_output_stride)
        features = refine_by_decoder(
            features,
            end_points,
            decoder_height=decoder_height,
            decoder_width=decoder_width,
            decoder_use_separable_conv=model_options.decoder_use_separable_conv,
            model_variant=model_options.model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)

    outputs_to_logits = {}
    for output in sorted(model_options.outputs_to_num_classes):
        outputs_to_logits[output] = _get_branch_logits(
            features,
            model_options.outputs_to_num_classes[output],
            model_options.atrous_rates,
            aspp_with_batch_norm=model_options.aspp_with_batch_norm,
            kernel_size=model_options.logits_kernel_size,
            weight_decay=weight_decay,
            reuse=reuse,
            scope_suffix=output)

    return outputs_to_logits


def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=tf.AUTO_REUSE,
                      is_training=False,
                      fine_tune_batch_norm=False):
    """Adds the decoder to obtain sharper segmentation results.

    Args:
      features: A tensor of size [batch, features_height, features_width,
        features_channels].
      end_points: A dictionary from components of the network to the corresponding
        activation.
      decoder_height: The height of decoder feature maps.
      decoder_width: The width of decoder feature maps.
      decoder_use_separable_conv: Employ separable convolution for decoder or not.
      model_variant: Model variant for feature extraction.
      weight_decay: The weight decay for model variables.
      reuse: Reuse the model variables or not.
      is_training: Is training or not.
      fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

    Returns:
      Decoder output with size [batch, decoder_height, decoder_width,
        decoder_channels].
    """
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'eps': 1e-5,
        'affine': True,
    }
    regularize_func = regularizer('l2', weight_decay)
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        with arg_scope([sep_conv2d], activate=tf.nn.relu, activate_middle=tf.nn.relu,
                       batch_norm=True, depthwise_weight_reg=None, pointwise_weight_reg=regularize_func,
                       padding='SAME', strides=[1, 1]):
            with arg_scope([conv2d], activate=tf.nn.relu, weight_reg=regularize_func,
                           batch_norm=True, padding='SAME', strides=[1, 1]):
                with arg_scope([batch_norm2d], **batch_norm_params):
                    with tf.variable_scope(_DECODER_SCOPE, _DECODER_SCOPE, [features]):
                        feature_list = feature_extractor.networks_to_feature_maps[
                            model_variant][feature_extractor.DECODER_END_POINTS]
                        if feature_list is None:
                            tf.logging.info('Not found any decoder end points.')
                            return features
                        else:
                            decoder_features = features
                            for i, name in enumerate(feature_list):
                                decoder_features_list = [decoder_features]

                                suffix = list(end_points.keys())[0].split('/')[0]
                                feature_name = '{}/{}'.format(
                                    suffix, name)
                                # [1, 1] to reduce channel to 4
                                decoder_features_list.append(
                                    conv2d(
                                        inputs=end_points[feature_name],
                                        outc=48,
                                        ksize=[1, 1],
                                        name='feature_projection' + str(i)))
                                # Resize to decoder_height/decoder_width.
                                for j, feature in enumerate(decoder_features_list):
                                    decoder_features_list[j] = tf.image.resize_bilinear(
                                        feature, [decoder_height, decoder_width], align_corners=True)
                                    decoder_features_list[j].set_shape(
                                        [None, decoder_height, decoder_width, None])
                                decoder_depth = 256
                                if decoder_use_separable_conv:
                                    # [3,3] kernel
                                    decoder_features = sep_conv2d(
                                        inputs=tf.concat(decoder_features_list, 3),
                                        ksize=[3, 3],
                                        outc=decoder_depth,
                                        ratios=[1, 1],
                                        name='decoder_conv0')
                                    decoder_features = sep_conv2d(
                                        inputs=decoder_features,
                                        ksize=[3, 3],
                                        outc=decoder_depth,
                                        ratios=[1, 1],
                                        name='decoder_conv1')
                                    DEBUG_VARS.decoder_features = decoder_features
                                else:
                                    decoder_features = conv2d(
                                        inputs=tf.concat(decoder_features_list, 3),
                                        outc=[decoder_depth],
                                        ksize=[3, 3],
                                        name='decoder_conv0')
                                    decoder_features = conv2d(
                                        inputs=decoder_features,
                                        outc=[decoder_depth],
                                        ksize=[3, 3],
                                        name='decoder_conv0')
                            return decoder_features


def _get_branch_logits(features,
                       num_classes,
                       atrous_rates=None,
                       aspp_with_batch_norm=False,
                       kernel_size=1,
                       weight_decay=0.0001,
                       reuse=tf.AUTO_REUSE,
                       scope_suffix=''):
    """Gets the logits from each model's branch.

    The underlying model is branched out in the last layer when atrous
    spatial pyramid pooling is employed, and all branches are sum-merged
    to form the final logits.

    Args:
      features: A float tensor of shape [batch, height, width, channels].
      num_classes: Number of classes to predict.
      atrous_rates: A list of atrous convolution rates for last layer.
      aspp_with_batch_norm: Use batch normalization layers for ASPP.
      kernel_size: Kernel size for convolution.
      weight_decay: Weight decay for the model variables.
      reuse: Reuse model variables or not.
      scope_suffix: Scope suffix for the model variables.

    Returns:
      Merged logits with shape [batch, height, width, num_classes].

    Raises:
      ValueError: Upon invalid input kernel_size value.
    """
    # When using batch normalization with ASPP, ASPP has been applied before
    # in _extract_features, and thus we simply apply 1x1 convolution here.
    if aspp_with_batch_norm or atrous_rates is None:
        if kernel_size != 1:
            raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                             'using aspp_with_batch_norm. Gets %d.' % kernel_size)
        atrous_rates = [1]

    with arg_scope(
            [conv2d],
            weight_reg=regularizer('l2', weight_decay),
            weight_init=tf.truncated_normal_initializer(stddev=0.01)):
        with tf.variable_scope(_LOGITS_SCOPE_NAME, _LOGITS_SCOPE_NAME, [features], reuse=reuse):
            branch_logits = []
            for i, rate in enumerate(atrous_rates):
                scope = scope_suffix
                if i:
                    scope += '_%d' % i

                branch_logits.append(
                    conv2d(
                        features,
                        outc=num_classes,
                        ksize=[kernel_size, kernel_size],
                        ratios=[rate, rate],
                        activate=None,
                        batch_norm=False,
                        use_bias=True,
                        name=scope))

            return tf.add_n(branch_logits)


def _build_deeplab(images, labels, ignore_label, FLAGS):
    """Builds a clone of DeepLab.

    Args:
      inputs_queue: A prefetch queue for images and labels.
      outputs_to_num_classes: A map from output type to the number of classes.
        For example, for the task of semantic segmentation with 21 semantic
        classes, we would have outputs_to_num_classes['semantic'] = 21.
      ignore_label: Ignore label.

    Returns:
      A map of maps from output_type (e.g., semantic prediction) to a
        dictionary of multi-scale logits names to logits. For each output_type,
        the dictionary has keys which correspond to the scales and values which
        correspond to the logits. For example, if `scales` equals [1.0, 1.5],
        then the keys would include 'merged_logits', 'logits_1.00' and
        'logits_1.50'.
    """

    outputs_to_scales_to_logits = multi_scale_logits(
        images,
        model_options=FLAGS,
        image_pyramid=FLAGS.image_pyramid,
        weight_decay=FLAGS.weight_decay,
        is_training=False,
        fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

    loss = None
    for output, num_classes in outputs_to_scales_to_logits.items():
        loss = ms_softmax_with_logits(
            outputs_to_scales_to_logits[output],
            labels,
            ignore_label,
            upsample_logits=FLAGS.upsample_logits,
            scope=output)

    return outputs_to_scales_to_logits, loss


if __name__ == '__main__':
    FLAGS = DeepLabFlags()
    # np.save('inputs_v.npy', inputs_v)
    with tf.device('/CPU:0'):
        inputs = tf.placeholder(name='inputs', shape=[16, 513, 513, 3], dtype=tf.float32)
        labels = tf.placeholder(name='labels', shape=[16, 513, 513, 1], dtype=tf.int32)
        outputs_to_scales_to_logits, loss = _build_deeplab(inputs, labels, [], FLAGS)

    inputs_v = np.random.rand(16, 513, 513, 3)
    labels_v = np.random.randint(0, 20, [16, 513, 513, 1])
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, save_path='/home/yifeng/Models/pretrain/SEGS/xception_voc_trainval/model.ckpt')

    outv = sess.run(loss, feed_dict={inputs: inputs_v, labels: labels_v})

    # print(np.mean(inputs_v), np.var(inputs_v))
    # def print_dict(d):
    #     if type(d) is dict:
    #         for i in d.keys():
    #             print(i)
    #             print_dict(d[i])
    #     elif type(d) is list:
    #         for i in d:
    #             print_dict(i)
    #     else:
    #         print(np.mean(d), np.var(d), np.max(d), np.min(d))
    # print_dict(outv)
    # print(outputs_to_scales_to_logits)
