#!/usr/bin/env python
# coding=utf8

import glob
import numpy as np
import os.path as osp
import pickle
import time
from sklearn.preprocessing import StandardScaler

from configure import default_feature_cfg

from feature_utils import extract_train_features

dataset_dir = osp.join(osp.dirname(osp.abspath(__file__)), "dataset")
feature_label_cache_path = osp.join(dataset_dir, "dataset.p")
scaler_cache_path = osp.join(dataset_dir, "feature_scaler.p")

feature_cfg = default_feature_cfg


def load_features_in_dir(dir_path, img_suffix=".png"):
    image_path_list = glob.glob(osp.join(dir_path, "*" + img_suffix))
    features = extract_train_features(image_path_list, feature_cfg)
    return features


# load all features and labels, normalizing features with StandardScaler,
# return features, labels and the feature scaler
def load_dataset(force_reload=False):
    if not force_reload and osp.isfile(feature_label_cache_path):
        with open(feature_label_cache_path, "rb") as f:
            dataset = pickle.load(f)
        with open(scaler_cache_path, "rb") as f:
            feature_scaler = pickle.load(f)
        print("using cached features, labels and feature-scaler")
        return dataset["features"], dataset["labels"], feature_scaler

    t_start = time.time()
    vehicle_dirs_name = ["GTI_Far", "GTI_Left", "GTI_MiddleClose", "GTI_RIGHT", "KITTI_extracted"]
    vehicle_dirs_path = [osp.join(dataset_dir, "vehicles", dir_name) for dir_name in vehicle_dirs_name]
    vehicle_features = []
    for vehicle_dir in vehicle_dirs_path:
        features = load_features_in_dir(vehicle_dir)
        vehicle_features.extend(features)
    num_vehicle_imgs = len(vehicle_features)
    print("extract features for %d vehicle images takes %.2fs" % (num_vehicle_imgs, time.time() - t_start))
    print("vehicle feature len: %d" % len(vehicle_features[0]))

    t_start = time.time()
    non_vehicle_dirs_name = ["GTI", "Extras"]
    non_vehicle_dirs_path = [osp.join(dataset_dir, "non-vehicles", dir_name) for dir_name in non_vehicle_dirs_name]
    non_vehicle_features = []
    for non_vehicle_dir in non_vehicle_dirs_path:
        features = load_features_in_dir(non_vehicle_dir)
        non_vehicle_features.extend(features)
    num_non_vehicle_imgs = len(non_vehicle_features)
    print("extract features for %d non-vehicle images takes %.2fs" % (num_non_vehicle_imgs, time.time() - t_start))
    print("non-vehicle feature len: %d" % len(non_vehicle_features[0]))

    t_start = time.time()
    all_features = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
    print("all_features.shape %s" %(all_features.shape, ))
    # normalize the features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(all_features)
    print("normalizeing features takes %.2fs" % (time.time() - t_start,))

    all_labels = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
    print("all_labels.shape %s" % (all_labels.shape, ))
    dataset = {
        "features": scaled_features,
        "labels": all_labels
    }

    with open(feature_label_cache_path, "wb") as f:
        pickle.dump(dataset, f)
    with open(scaler_cache_path, "wb") as f:
        pickle.dump(feature_scaler, f)

    return scaled_features, all_labels, feature_scaler


def load_feature_scaler():
    with open(scaler_cache_path, "rb") as f:
        feature_scaler = pickle.load(f)
    return feature_scaler
