#!/usr/bin/env python
# coding=utf8


class FeatureConfig(object):
    def __init__(self):
        self.hog_feature_cfg = {
            "color_space": "HLS",
            "orient": 9,
            "pix_per_cell": 8,
            "cells_per_block": 2,
            # list of channels to get the hog features
            "channels": [0, 1, 2],
        }

        self.color_feature_cfg = {
            "color_space": "HLS",
            # resize the image to this target size and then extract color features
            "target_size": (16, 16),
            # list of channels to get color features
            "channels": [0, 1, 2],
        }

        self.hist_feature_cfg = {
            "color_space": "HLS",
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

sliding_window_cfg = {
    "scales": (1.3, 2.0, 3.5),
    "y_start_stop": ((360, 520), (400, 560), (400, 660)),
    "x_start_stop": ((650, 1281), (650, 1281), (650, 1281), ),
    "box_color": (),
}

default_feature_cfg = FeatureConfig()
