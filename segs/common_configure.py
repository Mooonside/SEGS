import tensorflow as tf


def jupyter_flag(model_variant='xception_65'):
    class flags:
        pass

    flags.crop_size = [513, 513]
    flags.outputs_to_num_classes = [['semantic', 21]]
    flags.logits_kernel_size = 1
    flags.atrous_rates = [6, 12, 18]
    flags.output_stride = 16
    flags.model_variant = model_variant
    flags.image_pyramid = None
    flags.add_image_level_feature = True
    flags.aspp_with_batch_norm = True
    flags.aspp_with_separable_conv = True
    flags.multi_grid = None
    flags.depth_multiplier = 1
    flags.decoder_output_stride = 4
    flags.decoder_use_separable_conv = True
    flags.merge_method = 'max'
    flags.weight_decay = 5e-4
    flags.fine_tune_batch_norm = True
    flags.upsample_logits = True
    return flags


def DeepLabFlags(flags=None):
    if flags is None:
        flags = tf.app.flags

    # Flags for input preprocessing.
    flags.DEFINE_multi_integer('crop_size', [513, 513], 'Desired crop size.')

    flags.DEFINE_list('outputs_to_num_classes', [['semantic', 21]], '#classes of different tasks')

    # Model dependent flags.
    flags.DEFINE_integer('logits_kernel_size', 1,
                         'The kernel size for the convolutional kernel that '
                         'generates logits.')

    # For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
    # rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
    # one could use different atrous_rates/output_stride during training/evaluation.
    flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18], 'atrous_rates for aspp')

    flags.DEFINE_integer('output_stride', 16,
                         'The ratio of input to output spatial resolution.')

    # When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
    # When using 'xception_65', we set atrous_rates = [6, 12, 18] (output stride 16)
    # and decoder_output_stride = 4.
    flags.DEFINE_enum('model_variant', 'xception_65',
                      ['xception_65', 'mobilenet_v2'], 'DeepLab model variant.')

    flags.DEFINE_multi_float('image_pyramid', None,
                             'Input scales for multi-scale feature extraction.')

    flags.DEFINE_boolean('add_image_level_feature', True,
                         'Add image level feature.')

    flags.DEFINE_boolean('aspp_with_batch_norm', True,
                         'Use batch norm parameters for ASPP or not.')

    flags.DEFINE_boolean('aspp_with_separable_conv', True,
                         'Use separable convolution for ASPP or not.')

    flags.DEFINE_multi_integer('multi_grid', None,
                               'Employ a hierarchy of atrous rates for ResNet.')

    flags.DEFINE_float('depth_multiplier', 1.0,
                       'Multiplier for the depth (number of channels) for all '
                       'convolution ops used in MobileNet.')

    # For `xception_65`, use decoder_output_stride = 4. For `mobilenet_v2`, use
    # decoder_output_stride = None.
    flags.DEFINE_integer('decoder_output_stride', 4,
                         'The ratio of input to output spatial resolution when '
                         'employing decoder to refine segmentation results.')

    flags.DEFINE_boolean('decoder_use_separable_conv', True,
                         'Employ separable convolution for decoder or not.')

    flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'],
                      'Scheme to merge multi scale features.')

    # training configures
    flags.DEFINE_float('weight_decay', 0.00004,
                       'The value of the weight decay for training.')

    # Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
    # Set to False and use small batch size to save GPU memory.
    flags.DEFINE_boolean('fine_tune_batch_norm', False,
                         'Fine tune the batch norm parameters or not.')

    flags.DEFINE_boolean('upsample_logits', True,
                         'whether upsample logits to compute loss. If set False, then labels are downsanpled')
    return flags
