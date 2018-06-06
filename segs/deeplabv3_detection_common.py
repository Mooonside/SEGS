"""
IN YOLOV3, uses 3 layers, respectively downsample 32, downsample 16 and downsample 8
IN DEEOLABV3+, the nwetwork output layers if stride 16, so need to add more layer to generate downsample 32!
"""
import numpy as np

detection_feature_layers = [
    # downsample 8
    'xception_65/entry_flow/block2/unit_1/xception_module/add:0',
    # downsample 16
    'xception_65/middle_flow/block1/unit_16/xception_module/add:0',
    # downsample 32
    'xception_65/detection_branch/exit_flow/block3/unit_1/xception_module/separable_conv3/pointwise_conv/Relu:0'
]

detection_feature_strides = np.asarray([
    8,
    16,
    32
])

detection_anchors = np.asarray([
    [
        [0.02403846, 0.03125],
        [0.03846154, 0.07211538],
        [0.07932692, 0.05528846]
    ],
    [
        [0.07211538, 0.14663462],
        [0.14903846, 0.10817308],
        [0.14182692, 0.28605769]
    ],
    [
        [0.27884615, 0.21634615],
        [0.375, 0.47596154],
        [0.89663462, 0.78365385]
    ]
])
