import tensorflow as tf
from numpy import ceil

from deeplabv3_det import _get_logits
from segs.common_configure import jupyter_flag
from tf_ops.wrap_ops import tensor_shape


def get_image_pyramids(images, scales, method=tf.image.ResizeMethod.BILINEAR, align_corners=True):
    _, h, w, _ = tensor_shape(images)
    scales = sorted(scales)
    image_pyramids = []

    for scale in scales:
        scale_h, scale_w = int(ceil(scale * h)), int(ceil(scale * w))
        resize_image = tf.image.resize_images(images,
                                              size=[scale_h, scale_w],
                                              method=tf.image.ResizeMethod.BILINEAR,
                                              align_corners=align_corners)
        image_pyramids.append(resize_image)
    return image_pyramids


def multi_scale_infer(images, scales, infer_func, infer_func_args=None):
    image_pyramids = get_image_pyramids(images, scales=scales)
    logits_pyramids = []
    for resize_image in image_pyramids:
        logits = infer_func(resize_image, **infer_func_args)
        logits_pyramids.append(logits)
    return logits_pyramids


def multi_scale_loss(logits_pyramids, labels, loss_func, resize_labels=False, loss_func_args=None):
    _, label_h, label_w, _ = tensor_shape(labels)
    loss_pyramids = []
    for logits in logits_pyramids:
        _, h, w, _ = tensor_shape(logits)
        if resize_labels:
            resized_labels = tf.image.resize_nearest_neighbor(labels, size=[h, w], align_corners=True)
            loss = loss_func(logits, resized_labels, **loss_func_args)
        else:
            resized_logits = tf.image.resize_bilinear(logits, size=[label_h, label_w], align_corners=True)
            loss = loss_func(resized_logits, labels, **loss_func_args)
        loss_pyramids.append(loss)
    return loss_pyramids


def scale_sensitive_loss(logits, labels, valid_scale_range=(0.0, 1.0)):
    pass
    return


if __name__ == '__main__':
    images = tf.get_variable(shape=[16, 513, 513, 3], dtype=tf.float32, name='images')
    image_pyramids = get_image_pyramids(images, scales=[0.5, 1.0, 1.5])
    infer_func_args = {
        'model_options': jupyter_flag(),
        'outputs_to_num_classes': {
            'semantics': 21
        },
        'weight_decay': 0.0001,
        'reuse': tf.AUTO_REUSE,
        'is_training': False,
        'fine_tune_batch_norm': False
    }
    logits_pyramids = multi_scale_infer(images, scales=[1.0, 2.0], infer_func=_get_logits,
                                        infer_func_args=infer_func_args)
    print(logits_pyramids)
