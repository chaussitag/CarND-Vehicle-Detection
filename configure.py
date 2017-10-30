#!/usr/bin/env python
# coding=utf8


class FeatureConfig(object):
    def __init__(self):
        self.hog_feature_cfg = {
            "color_space": "YCrCb",
            "orient": 11,
            "pix_per_cell": 16,
            "cells_per_block": 2,
            # list of channels to get the hog features
            "channels": [0, 1, 2],
        }

        self.color_feature_cfg = {
            "color_space": "YCrCb",
            # resize the image to this target size and then extract color features
            "target_size": (32, 32),
            # list of channels to get color features
            "channels": [0, 1, 2],
        }

        self.hist_feature_cfg = {
            "color_space": "YCrCb",
            "nbins": 32,
            "channels": [0, 1, 2],
        }

        self.use_color_feature = True
        self.use_hist_feature = True
        self.use_hog_feature = True


# sliding_window_cfg = {
#     "scales": (1.0, 1.5, 2.0, 3.5),
#     "y_start_stop": ((400, 485), (400, 528), (400, 560), (400, 660)),
#     "x_start_stop": ((620, 1281), (620, 1281), (620, 1281), (620, 1281), ),
#     "box_color": (),
# }

# sliding_window_cfg = {
#     "scales": (
#         0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8,
#         2.0, 2.25, 2.5, 2.8,
#         3.0, 3.25, 3.5
#     ),
#     "y_start_stop": (
#         (320, 580), (320, 580), (320, 580), (320, 580), (320, 580), (340, 600), (340, 600), (340, 600), (340, 620),
#         (320, 620), (320, 620), (320, 640), (360, 640),
#         (420, 640), (420, 640), (420, 660),
#     ),
#     "x_start_stop": (
#         (800, 1080), (800, 1080), (820, 1180), (820, 1200), (820, 1181), (640, 1221), (640, 1281), (640, 1281), (640, 1281),
#         (650, 1281), (650, 1281), (660, 1281), (670, 1281),
#         (660, 1281), (660, 1281), (680, 1281),
#     ),
#     "box_color": (),
# }

sliding_window_cfg = {
    "scales": (
        0.8, 0.9, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8,
        2.0, 2.25, 2.5, 2.8,
        3.0, 3.25, 3.5
    ),
    "y_start_stop": (
        (320, 580), (320, 580), (320, 580), (320, 580), (340, 600), (340, 600), (340, 600), (340, 620),
        (320, 620), (320, 620), (320, 640), (360, 640),
        (420, 640), (420, 640), (420, 660),
    ),
    # "y_start_stop": (
    #     (320, 520), (320, 520), (320, 560), (360, 580), (360, 600), (400, 600), (400, 600), (400, 620),
    #     (400, 620), (400, 620), (400, 640), (400, 640),
    #     (420, 640), (420, 640), (420, 660),
    # ),
    "x_start_stop": (
        (800, 1080), (820, 1180), (820, 1200), (820, 1181), (640, 1221), (640, 1281), (640, 1281), (640, 1281),
        (650, 1281), (650, 1281), (660, 1281), (670, 1281),
        (660, 1281), (660, 1281), (680, 1281),
    ),
    "box_color": (),
}

default_feature_cfg = FeatureConfig()
