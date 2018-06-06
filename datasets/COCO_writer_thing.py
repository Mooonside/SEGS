from __future__ import division

import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('/mnt/disk50/datasets/COCO/cocoapi/PythonAPI')

from pycocotools.coco import COCO
from COCO_writer_stuff import int64_feature, bytes_feature, DEFAULT_PATHS


def real_id_to_cat_id(realId):
    """
    Note coco has 80 classes, but the catId ranges from 1 to 90!
    Args:
        catId: id for our training set.
    Returns:
        real id in coco datasets.
    """
    real_id_to_cat_id = \
    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
     17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34,
     31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48, 44: 49,
     45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63,
     59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81,
     73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}
    return real_id_to_cat_id[realId]


def cat_id_to_real_id(catId):
    """
    Note coco has 80 classes, but the catId ranges from 1 to 90!
    Args:
        realId: id in coco datasets.
    Returns:
        id for our training set.
    """
    cat_id_to_real_id = \
    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
     18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
     35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
     50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
     64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
     82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
    return cat_id_to_real_id[catId]


def coco_cat_id_to_voc_id(catId):
    """
    Note coco has 80 classes, but pascal voc has 20 classes.
    So this function convert coco catId to voc id. 60 classes rest are assigned to background.
    Return:
         id for voc.
    """
    realId = cat_id_to_real_id(catId)
    real_id_to_voc_id = \
    {1: 15, 2: 2, 3: 7, 4: 14, 5: 1, 6: 6, 7: 19, 8: 0, 9: 4, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 3, 16: 8, 17: 12,
     18: 13, 19: 17, 20: 10, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0,
     35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 5, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0,
     52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 9, 58: 18, 59: 16, 60: 0, 61: 11, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0,
     69: 20, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0}
    return real_id_to_voc_id[realId]



class COCOWriter:
    """
        COCO detection challenge Datasets to TF record Writer.
    """

    def __init__(self, paths=DEFAULT_PATHS):
        self.img_path = paths['images']
        self.ano_path = paths['annotations']

    def get_coco_gt(self, coco, img_id, height, width, img_name):
        """
            get the masks, bboxes and cls for all the instances
        Note: some images are not annotated
        Return:
             masks, mxhxw numpy array
             bboxes, mx4
             classes, mx1
        """
        annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        # assert annIds is not None and annIds > 0, 'No annotation for %s' % str(img_id)
        anns = coco.loadAnns(annIds)
        # assert len(anns) > 0, 'No annotation for %s' % str(img_id)
        masks = []
        classes = []
        bboxes = []

        for ann in anns:
            id = cat_id_to_real_id(ann['category_id'])
            # id = coco_cat_id_to_voc_id(ann['category_id'])
            if id != 0:
                classes.append(id)

                m = coco.annToMask(ann) # {0, 1} mask
                assert m.shape[0] == height and m.shape[1] == width, \
                    'image %s and ann %s don''t match' % (img_id, ann)
                masks.append(m)

                bboxes.append(ann['bbox'])

        masks = np.asarray(masks)
        classes = np.asarray(classes)
        bboxes = np.asarray(bboxes)

        # to x1, y1, x2, y2
        num_classes = bboxes.shape[0]
        if num_classes <= 0:
            bboxes = np.zeros([0, 4], dtype=np.float32)
            classes = np.zeros([0], dtype=np.float32)
            num_classes = 0
            print('None Annotations %s' % img_name)
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 0] + bboxes[:, 3]

        bboxes = bboxes.astype(np.float32)
        classes = classes.astype(np.float32)
        masks = masks.astype(np.uint8)
        assert masks.shape[0] == bboxes.shape[0], 'Shape Error'

        return num_classes, masks, bboxes, classes

    def convert_to_example(self, img_name, img_data, img_format, num, masks, bboxes, classes, height, width):
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/name': bytes_feature(img_name.encode()),
            'image/height': int64_feature(int(height)),
            'image/width': int64_feature(int(width)),
            'image/channels': int64_feature(int(3)),
            'image/format': bytes_feature(img_format),
            'image/encoded': bytes_feature(img_data),
            'label/num_classes': int64_feature(num),
            'label/masks': bytes_feature(masks.tostring()),
            'label/bboxes': bytes_feature(bboxes.tostring()),
            'label/classes': bytes_feature(classes.tostring())
        }))
        return example

    def add_to_record(self, img_name, img_data, img_format, num, masks, bboxes, classes, height, width, tfrecord_writer):
        example = self.convert_to_example(img_name, img_data, img_format, num, masks, bboxes, classes, height, width)
        tfrecord_writer.write(example.SerializeToString())

    def run(self, record_dir, split_name, size=1000):
        ano_path = os.path.join(self.ano_path, split_name, 'instances_%s.json' % (split_name))
        record_dir = os.path.join(record_dir, split_name)
        coco_ins = COCO(ano_path)
        print('%s has %d images' % (split_name, len(coco_ins.imgs)))
        imgs = [(img_id, coco_ins.imgs[img_id]) for img_id in coco_ins.imgs]

        total_num = len(imgs)
        for start in range(0, total_num, size):
            tf_filename = '%s/%03d.tfrecord' % (record_dir, start // size)
            tf_recorder = tf.python_io.TFRecordWriter(tf_filename)
            print('=>' * (start * 5 // total_num) + '{:.0f}% Finished'.format(start / total_num * 100))
            for pic_idx in range(start, min(start + size, total_num)):
                img_id = imgs[pic_idx][0]
                img_name = imgs[pic_idx][1]['file_name']
                img_path = os.path.join(self.img_path, split_name, img_name)
                img_data = tf.gfile.FastGFile(img_path, 'rb').read()
                img_format = b'JPEG'
                height = imgs[pic_idx][1]['height']
                width = imgs[pic_idx][1]['width']
                num_classes, masks, bboxes, classes = self.get_coco_gt(coco_ins, img_id, height, width, img_name)
                if num_classes == 0:
                    print('skip image %s' % (img_id))
                    continue
                else:
                    self.add_to_record(img_name, img_data, img_format, num_classes,
                                   masks, bboxes, classes, height, width, tf_recorder)

        print('=>' * 5 + '{:.0f}% Finished'.format(100))


def convert_to_tfrecord(record_dir, split_name, size):
    writer = COCOWriter()
    writer.run(record_dir, split_name, size)



if __name__ == '__main__':
    split_name = 'train2017'
    record_dir = '/mnt/disk50/datasets/COCO/tf_records/detection/coco'
    convert_to_tfrecord(record_dir, split_name, size=1000)