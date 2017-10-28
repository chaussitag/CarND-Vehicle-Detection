#!/usr/bin/env python
# coding=utf8

import glob
import numpy as np
import os.path as osp
import pickle
import time
from sklearn.preprocessing import StandardScaler

from feature_utils import extract_train_features

dataset_dir = osp.join(osp.dirname(osp.abspath(__file__)), "dataset")
feature_label_cache_path = osp.join(dataset_dir, "dataset.p")
scaler_cache_path = osp.join(dataset_dir, "feature_scaler.p")


def load_features_in_dir(dir_path, img_suffix=".png"):
    image_path_list = glob.glob(osp.join(dir_path, "*" + img_suffix))
    features = extract_train_features(image_path_list, color_space='HSV', spatial_size=(32, 32),
                                      hist_bins=32, orient=9,
                                      pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
                                      color_feature=True, hist_feature=True, hog_feature=True)
    return features


# load all features and labels, normalizing features with StandardScaler,
# return features, labels and the feature scaler
def load_dataset(force_reload=False):
    if not force_reload and osp.isfile(feature_label_cache_path):
        with open(feature_label_cache_path, "rb") as f:
            dataset = pickle.load(f)
        with open(scaler_cache_path, "rb") as f:
            feature_scaler = pickle.load(f)
        print("using cached features, labels and feature-sacler")
        return dataset["features"], dataset["labels"], feature_scaler

    t_start = time()
    vehicle_dirs_name = ["GTI_Far", "GTI_Left", "GTI_MiddleClose", "GTI_RIGHT", "KITTI_extracted"]
    vehicle_dirs_path = [osp.join(dataset_dir, "vehicles", dir_name) for dir_name in vehicle_dirs_name]
    vehicle_features = []
    for vehicle_dir in vehicle_dirs_path:
        features = load_features_in_dir(vehicle_dir)
        vehicle_features.extend(features)
    print("extract features for vehicle images takes %.2fs" % (time() - t_start, ))

    t_start = time()
    non_vehicle_dirs_name = ["GTI", "Extras"]
    non_vehicle_dirs_path = [osp.join(dataset_dir, "non-vehicles", dir_name) for dir_name in non_vehicle_dirs_name]
    non_vehicle_features = []
    for non_vehicle_dir in non_vehicle_dirs_path:
        features = load_features_in_dir(non_vehicle_dir)
        non_vehicle_features.extend(features)
    print("extract features for non-vehicle images takes %.2fs" % (time() - t_start,))

    t_start = time()
    all_features = np.concatenate((vehicle_features, non_vehicle_features))
    # normalize the features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(all_features)
    print("normalizeing features takes %.2fs" % (time() - t_start,))

    all_labels = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
    dataset = {
        "features": scaled_features,
        "labels": all_labels
    }

    with open(feature_label_cache_path, "wb") as f:
        pickle.dump(dataset, f)
    with open(scaler_cache_path, "wb") as f:
        pickle.dump(feature_scaler)

    return scaled_features, all_labels, feature_scaler


def load_feature_scaler():
    with open(scaler_cache_path, "rb") as f:
        feature_scaler = pickle.load(f)
    return feature_scaler
