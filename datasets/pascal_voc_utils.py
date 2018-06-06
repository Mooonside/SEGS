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
    'background',  # 0
    'aeroplane',  # 1
    'bicycle',  # 2
    'bird',  # 3
    'boat',  # 4
    'bottle',  # 5
    'bus',  # 6
    'car',  # 7
    'cat',  # 8
    'chair',  # 9
    'cow',  # 10
    'diningtable',  # 11
    'dog',  # 12
    'horse',  # 13
    'motorbike',  # 14
    'person',  # 15
    'pottedplant',  # 16
    'sheep',  # 17
    'sofa',  # 18
    'train',  # 19
    'tvmonitor'  # 20
]

pascal_voc_dict = dict([[c, idx] for idx, c in enumerate(pascal_voc_classes)])
