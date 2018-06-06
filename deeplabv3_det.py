import tensorflow as tf

from backbones import feature_extractor
from segs.common_configure import jupyter_flag
from segs.deeplabv3_detection_common import detection_feature_layers, detection_anchors
from tf_ops.wrap_ops import conv2d, sep_conv2d, drop_out, arg_scope, regularizer, \
    batch_norm2d, avg_pool2d, tensor_shape

_LOGITS_SCOPE_NAME = 'logits'
_DETECTION_SCOPE_NAME = 'detection_logits'
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
                            _, image_h, image_w, _ = tensor_shape(images)
                            pool_height = scale_dimension(image_h,
                                                          1. / model_options.output_stride)
                            pool_width = scale_dimension(image_w,
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
                        concat_logits = conv2d(inputs=concat_logits, outc=depth, ksize=[1, 1],
                                               name=_CONCAT_PROJECTION_SCOPE)
                        DEBUG_VARS.aspp_concat_feature = concat_logits
                        concat_logits = drop_out(concat_logits, kp_prob=0.9, is_training=is_training,
                                                 name=_CONCAT_PROJECTION_SCOPE + '_dropout')

                        return concat_logits, end_points


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


def _get_logits(images,
                model_options,
                outputs_to_num_classes,
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
    print('ASPP FEATRUES', features)
    for i in end_points.keys():
        print(end_points[i])

    DEBUG_VARS.aspp_result = features
    if model_options.decoder_output_stride is not None:
        _, image_h, image_w, _ = tensor_shape(images)
        decoder_height = scale_dimension(image_h,
                                         1.0 / model_options.decoder_output_stride)
        decoder_width = scale_dimension(image_w,
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
    for output in sorted(outputs_to_num_classes):
        outputs_to_logits[output] = _get_branch_logits(
            features,
            outputs_to_num_classes[output],
            model_options.atrous_rates,
            aspp_with_batch_norm=model_options.aspp_with_batch_norm,
            kernel_size=model_options.logits_kernel_size,
            weight_decay=weight_decay,
            reuse=reuse,
            scope_suffix=output)

        outputs_to_logits['detection'] = _get_detection(
            end_points,
            num_classes=outputs_to_num_classes[output],
            weight_decay=weight_decay,
            reuse=reuse,
            scope_suffix='scale'
        )

    return outputs_to_logits


def _get_detection(end_points,
                   num_classes,
                   weight_decay=0.0001,
                   reuse=tf.AUTO_REUSE,
                   scope_suffix=''
                   ):
    with arg_scope(
        [conv2d],
        weight_reg=regularizer('l2', weight_decay),
        weight_init=tf.truncated_normal_initializer(stddev=0.01)):
        with tf.variable_scope(_DETECTION_SCOPE_NAME, _DETECTION_SCOPE_NAME, [end_points], reuse=reuse):
            branch_logits = []
            for layer_idx, layer_name in enumerate(detection_feature_layers):
                features = end_points[layer_name]
                branch_logits.append(
                    conv2d(
                        features,
                        outc=len(detection_anchors[layer_idx]) * (4 + num_classes),
                        ksize=[1, 1],
                        activate=None,
                        batch_norm=False,
                        use_bias=True,
                        name=scope_suffix if scope_suffix is None else (scope_suffix + '_%d' % layer_idx)))
            return branch_logits


if __name__ == '__main__':
    images = tf.get_variable(shape=[16, 513, 513, 3], dtype=tf.float32, name='images')
    logits = _get_logits(images=images,
                         model_options=jupyter_flag(model_variant='xception_65_detection'),
                         outputs_to_num_classes={
                             'semantics': 21
                         },
                         weight_decay=0.0001,
                         reuse=tf.AUTO_REUSE,
                         is_training=False,
                         fine_tune_batch_norm=False)
    print(logits)
