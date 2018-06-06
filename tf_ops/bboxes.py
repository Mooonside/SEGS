import tensorflow as tf
from tf_ops.wrap_ops import safe_divide
from tensorflow.python.ops import control_flow_ops


def bboxes_intersection(bboxes, bbox_ref, name=None):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
        tf.add_to_collection('debug_scores', scores)
        return scores


def bboxes_resize(bboxes, bbox_ref, name=None):
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v

        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def bboxs_clip(bboxes, vmin=0, vmax=1, name=None):
    with tf.name_scope(name, 'bboxes_clip'):
        bboxes = tf.unstack(bboxes, axis=-1)

        bboxes[0] = tf.maximum(bboxes[0], vmin)
        bboxes[1] = tf.maximum(bboxes[1], vmin)
        bboxes[2] = tf.minimum(bboxes[2], vmax)
        bboxes[3] = tf.minimum(bboxes[3], vmax)

        bboxes = tf.stack(bboxes, axis=-1)
    return bboxes


def filter_overlap(labels, bboxes, threshold):
    scores = bboxes_intersection(bboxes, tf.constant([0, 0, 1, 1], bboxes.dtype))
    mask = scores > threshold
    labels = tf.boolean_mask(labels, mask)
    bboxes = tf.boolean_mask(bboxes, mask)
    return labels, bboxes


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered,
                                aspect_ratio_range,
                                area_range,
                                max_attempts,
                                intersection_threshold=0.5,
                                clip=True,
                                scope=None):
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        # [y_min, x_min, y_max, x_max]
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = bboxes_resize(bboxes, distort_bbox)
        labels, bboxes = filter_overlap(labels, bboxes, threshold=intersection_threshold)

        if clip:
            bboxes = bboxs_clip(bboxes)
    return cropped_image, labels, bboxes, distort_bbox


def random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes.
    """
    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):

        uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = tf.less(uniform_random, .5)
        # Flip image.
        result = control_flow_ops.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
        # Flip bboxes.
        bboxes = control_flow_ops.cond(mirror_cond, lambda: flip_bboxes(bboxes), lambda: bboxes)
        result.set_shape(image.get_shape())
        return result, bboxes