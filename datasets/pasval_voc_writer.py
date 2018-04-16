import os
import random
import xml.etree.ElementTree as ET

import tensorflow as tf


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


DEFUALT_PATHS = {
    'images': '/mnt/disk/chenyifeng/VOC2012/JPEGImages',
    'annotations': '/mnt/disk/chenyifeng/VOC2012/Annotations',
    'segmentations': '/mnt/disk/chenyifeng/VOC2012/SegmentationClassAug'
}


class PascalVocWriter:
    """
        PASCAL VOC 2012 DataSet to TF record Writer
    """

    def __init__(self, paths=DEFUALT_PATHS):
        self.img_path = paths['images']
        self.ano_path = paths['annotations']
        self.sgm_path = paths['segmentations']
    def convert_to_example(self, file_name):
        img_path = os.path.join(self.img_path, file_name + '.jpg')
        ano_path = os.path.join(self.ano_path, file_name + '.xml')
        sgm_path = os.path.join(self.sgm_path, file_name + '.png')

        img_data = tf.gfile.FastGFile(img_path, 'rb').read()
        sgm_data = tf.gfile.FastGFile(sgm_path, 'rb').read()

        # img_data = imread(img_path).tostring()
        # sgm_data = imread(sgm_path).tostring()

        anno_tree = ET.parse(ano_path)
        anno_root = anno_tree.getroot()

        # is_sgmt = int(anno_root.find('segmented').text)
        # if is_sgmt == 0:
        #     print('{} is not a Segmentation Sample. So Skipped'.format(file_name))

        size = anno_root.find('size')
        shape = [int(size.find('height').text),
                 int(size.find('width').text),
                 int(size.find('depth').text)]

        image_format = b'JPEG'
        segment_format = b'PNG'

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/name':bytes_feature(file_name.encode()),
                    'image/height': int64_feature(shape[0]),
                    'image/width': int64_feature(shape[1]),
                    'image/channels': int64_feature(shape[2]),
                    'image/shape': int64_feature(shape),
                    'image/format': bytes_feature(image_format),
                    'image/encoded': bytes_feature(img_data),
                    'label/format': bytes_feature(segment_format),
                    'label/encoded': bytes_feature(sgm_data)
                }
            )
        )
        return example

    def add_to_record(self, file_name, tfrecord_writer):
        example = self.convert_to_example(file_name)
        tfrecord_writer.write(example.SerializeToString())

    def run(self, pic_names, output_dir, shuffling=False, size=300):
        if shuffling:
            random.seed(1314)
            random.shuffle(pic_names)

        total_num = len(pic_names)

        for start in range(0, total_num, size):
            tf_filename = '%s/%03d.tfrecord' % (output_dir, start // size)
            tf_recorder = tf.python_io.TFRecordWriter(tf_filename)
            print('=>' * (start * 5 // total_num) + '{:.0f}% Finished'.format(start / total_num * 100))
            for pic_idx in range(start, min(start + 300, total_num)):
                pic_name = pic_names[pic_idx]
                self.add_to_record(pic_name, tf_recorder)

        print('=>' * 5 + '{:.0f}% Finished'.format(100))


def convert_val():
    writer = PascalVocWriter()
    pic_names = open('/mnt/disk/chenyifeng/VOC2012/ImageSets/Segmentation/val.txt').readlines()
    pic_names = [i.strip(' \n') for i in pic_names]
    writer.run(pic_names, output_dir='/mnt/disk/chenyifeng/VOC2012/tf_segments/tf_records/val')


def convert_train():
    writer = PascalVocWriter()
    pic_names = open('/mnt/disk/chenyifeng/VOC2012/ImageSets/Segmentation/train.txt').readlines()
    pic_names = [i.strip(' \n') for i in pic_names]
    writer.run(pic_names, output_dir='/mnt/disk/chenyifeng/VOC2012/tf_segments/tf_records/train')


if __name__ == '__main__':
    # convert_train()
    convert_val()
