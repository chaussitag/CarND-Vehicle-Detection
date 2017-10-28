#!/usr/bin/env python
# coding=utf8

class FeatureConfig(object):
    def __init__(self):

        self.hog_feature_cfg = {
            "color_space" : "HSV",
            "orient": 9,
            "pix_per_cell": 8,
            "cells_per_block": 2,
            # list of channels to get the hog features
            "channels": [0, 1, 2],
        }

        self.color_feature_cfg = {
            "color_space": "HSV",
            # resize the image to this target size and then extract color features
            "target_size": (32, 32),
            # list of channels to get color features
            "channels": [0, 1, 2],
        }

        self.hist_feature_cfg = {
            "color_space": "HSV",
            "nbins" : 32,
            "channels": [0, 1, 2],
        }

        self.use_color_feature = True
        self.use_hist_feature = True
        self.use_hog_feature = True


default_feature_cfg = FeatureConfig()