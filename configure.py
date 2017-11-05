#!/usr/bin/env python
# coding=utf8


class FeatureConfig(object):
    def __init__(self):
        self.hog_feature_cfg = {
            "color_space": "YCrCb",
            "orient": 12,
            "pix_per_cell": 8,
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
#     "scales": (
#         1.0, 1.2, 1.5, 1.8,
#         2.0, 2.25, 2.5, 2.8,
#         3.0, 3.25,
#     ),
#     "y_start_stop": (
#         (360, 540), (360, 600), (360, 600), (370, 630),
#         (370, 640), (380, 650), (380, 650), (380, 650),
#         (380, 650), (380, 660),
#     ),
#
#
#     "x_start_stop": (
#         (650, 1281), (660, 1281), (640, 1281), (640, 1281),
#         (800, 1281), (800, 1281), (800, 1281), (800, 1281),
#         (800, 1281), (800, 1281),
#     ),
#     "box_color": (),
# }


sliding_window_cfg = {
    "scales": (
        1.0, 1.15, 1.3, 1.5, 1.8,
        2.0, 2.5, 2.8,
        3.0, 3.25,
    ),

    "y_start_stop": (
        (365, 540), (365, 600), (365, 600), (365, 600), (375, 630),
        (370, 640), (380, 650), (380, 650),
        (380, 650), (380, 660),
    ),

    "x_start_stop": (
        (650, 1281), (660, 1281), (660, 1281), (640, 1281), (640, 1281),
        (800, 1281), (800, 1281), (800, 1281),
        (800, 1281), (800, 1281),
    ),
    "box_color": (),
}

default_feature_cfg = FeatureConfig()
