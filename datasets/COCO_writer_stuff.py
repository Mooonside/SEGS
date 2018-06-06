from __future__ import division

import os
import  numpy as np
import tensorflow as tf
import sys
sys.path.append('/mnt/disk50/datasets/COCO/cocoapi/PythonAPI')

from pycocotools.coco import COCO


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


DEFAULT_PATHS = {
    'images': '/mnt/disk50/datasets/COCO/images/',
    'annotations': '/mnt/disk50/datasets/COCO/annotations',
}


class COCOWriter:
    """
        COCO 2017 stuff challenge Datasets to TF record Writer.
    """

    def __init__(self, paths=DEFAULT_PATHS):
        self.img_path = paths['images']
        self.ano_path = paths['annotations']

    def cocoAnoToSegMap(self, coco, imgId, checkUniquePixelLabel=True, includeCrowd=False):
        '''
            Convert COCO GT or results for a single image to a segmentation map.
            :param coco: an instance of the COCO API (ground-truth or result)
            :param imgId: the id of the COCO image
            :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
            :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
            :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
            '''

        # Init
        curImg = coco.imgs[imgId]
        imageSize = (curImg['height'], curImg['width'])
        labelMap = np.zeros(imageSize)

        # Get annotations of the current image (may be empty)
        if includeCrowd:
            annIds = coco.getAnnIds(imgIds=imgId)
        else:
            annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
        imgAnnots = coco.loadAnns(annIds)

        # Combine all annotations of this image in labelMap
        # labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
        for a in range(0, len(imgAnnots)):
            labelMask = coco.annToMask(imgAnnots[a]) == 1
            # labelMask = labelMasks[:, :, a] == 1
            newLabel = imgAnnots[a]['category_id']

            if checkUniquePixelLabel and (labelMap[labelMask] != 0).any():
                raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))

            labelMap[labelMask] = newLabel

        # set 0 in labelMap to 184
        labelMap[labelMap == 0] += 184
        # set label scope to [0, 92]
        labelMap -= 92
        return labelMap

    def convert_to_example(self, img_name, img_data, img_format, label_data, height, width):
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/name': bytes_feature(img_name.encode()),
            'image/height': int64_feature(int(height)),
            'image/width': int64_feature(int(width)),
            'image/channels': int64_feature(int(3)),
            'image/format': bytes_feature(img_format),
            'image/encoded': bytes_feature(img_data),
            'label/encoded': bytes_feature(label_data.tostring())
        }))
        return example

    def add_to_record(self, img_name, img_data, img_format, label_data, height, width, tfrecord_writer):
        example = self.convert_to_example(img_name, img_data, img_format, label_data, height, width)
        tfrecord_writer.write(example.SerializeToString())


    def run(self, record_dir, split_name, size=1000):
        ano_path = os.path.join(self.ano_path, split_name, 'stuff_%s.json' % (split_name))
        record_dir = os.path.join(record_dir, split_name)
        coco_stuff = COCO(ano_path)
        print('%s has %d images' % (split_name, len(coco_stuff.imgs)))
        imgs = [(img_id, coco_stuff.imgs[img_id]) for img_id in coco_stuff.imgs]

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
                label_data = self.cocoAnoToSegMap(coco_stuff, img_id)
                height, width = imgs[pic_idx][1]['height'], imgs[pic_idx][1]['width']
                self.add_to_record(img_name, img_data, img_format, label_data, height, width, tf_recorder)

        print('=>' * 5 + '{:.0f}% Finished'.format(100))

def convert_to_tfrecord(record_dir, split_name, size):
    writer = COCOWriter()
    writer.run(record_dir, split_name, size)


if __name__ == '__main__':
    split_name = 'val2017'
    record_dir = '/mnt/disk50/datasets/COCO/tf_records/stuff'
    convert_to_tfrecord(record_dir, split_name, size=1000)