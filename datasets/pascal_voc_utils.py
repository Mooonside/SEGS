import numpy as np

__all__ = [
    'pascal_voc_palette',
    'pascal_voc_classes'
]

pascal_voc_palette = np.array([[0, 0, 0],
                               [128, 0, 0],
                               [0, 128, 0],
                               [128, 128, 0],
                               [0, 0, 128],
                               [128, 0, 128],
                               [0, 128, 128],
                               [128, 128, 128],
                               [64, 0, 0],
                               [192, 0, 0],
                               [64, 128, 0],
                               [192, 128, 0],
                               [64, 0, 128],
                               [192, 0, 128],
                               [64, 128, 128],
                               [192, 128, 128],
                               [0, 64, 0],
                               [128, 64, 0],
                               [0, 192, 0],
                               [128, 192, 0],
                               [0, 64, 128]], dtype='uint8')

pascal_voc_classes = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]
