#!/usr/bin/env python
# coding=utf8

import numpy as np
import cv2
from skimage.feature import hog


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cells_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cells_per_block, cells_per_block),
                                  transform_sqrt=True, block_norm="L2-Hys",
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       transform_sqrt=True, block_norm="L2-Hys",
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def color_digest(img, target_size, channels):
    resized_img = cv2.resize(img, target_size)
    # Return the feature vector
    return resized_img[:, :, channels].ravel()


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins, channels, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    features = []
    for c in channels:
        hist = np.histogram(img[:, :, c], bins=nbins, range=bins_range)
        features.append(hist[0])
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(features)
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def convert_color(bgr_img, color_space='BGR'):
    if color_space != 'BGR':
        if color_space == 'HSV':
            converted_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            converted_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            converted_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            converted_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            converted_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
        elif color_space == "Gray":
            converted_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2GRAY)
            converted_img = np.expand_dims(converted_img, 2)
        else:
            converted_img = np.copy(bgr_img)
    else:
        converted_img = np.copy(bgr_img)

    return converted_img


# Define a function to extract features from a single image window
def single_window_features(img_window, feature_cfg):
    # 1) Define an empty list to receive features

    hog_feature_cfg = feature_cfg.hog_feature_cfg
    color_feature_cfg = feature_cfg.color_feature_cfg
    hist_feature_cfg = feature_cfg.hist_feature_cfg

    # 2) Apply color conversion if other than 'BGR'
    hog_color_space = hog_feature_cfg["color_space"]
    img_for_hog = convert_color(img_window, hog_color_space)

    img_features = []
    # 3) Compute spatial features if flag is set
    if feature_cfg.use_color_feature is True:
        if color_feature_cfg["color_space"] == hog_color_space:
            img_for_color = img_for_hog
        else:
            img_for_color = convert_color(img_window, color_feature_cfg["color_space"])
        color_features = color_digest(img_for_color, color_feature_cfg["target_size"], color_feature_cfg["channels"])
        # 4) Append features to list
        img_features.append(color_features)

    # 5) Compute histogram features if flag is set
    if feature_cfg.use_hist_feature is True:
        if hist_feature_cfg["color_space"] == hog_color_space:
            img_for_hist = img_for_hog
        else:
            img_for_hist = convert_color(img_window, hist_feature_cfg["color_space"])
        hist_features = color_hist(img_for_hist, hist_feature_cfg["nbins"], hist_feature_cfg["channels"])
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if feature_cfg.use_hog_feature is True:
        hog_features = []
        for channel in hog_feature_cfg["channels"]:
            hog_features.extend(get_hog_features(img_for_hog[:, :, channel],
                                                 hog_feature_cfg["orient"], hog_feature_cfg["pix_per_cell"],
                                                 hog_feature_cfg["cells_per_block"],
                                                 vis=False, feature_vec=True))
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function to extract features from a list of training images, all traing images have the same size.
# Have this function call bin_spatial() and color_digest()
def extract_train_features(train_image_list, feature_cfg):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for path in train_image_list:
        # Read in each one by one
        image = cv2.imread(path)
        file_features = single_window_features(image, feature_cfg)
        # assert file_features.shape[0] == 9636, "%s feature error" % path
        features.append(file_features)
    # Return list of feature vectors
    return features


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    #imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    #return imcopy
