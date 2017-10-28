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
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       transform_sqrt=True,
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
        features.append(np.histogram(img[:, :, c], bins=nbins, range=bins_range))
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
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def get_slide_windows(img, x_start_stop_=(None, None), y_start_stop_=(None, None),
                      xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop = list(x_start_stop_)
    y_start_stop = list(y_start_stop_)
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of get_slide_windows())
def search_windows(img, windows, clf, scaler, feature_cfg):
    # 1) Create an empty list to receive positive detection windows
    postive_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_window_features(test_img, feature_cfg)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            postive_windows.append(window)
    # 8) Return windows for positive detections
    return postive_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def sliding_window_search(img, ystart, ystop, scale, classifier, X_scaler, feature_cfg):
    # img = img.astype(np.float32) / 255

    roi = img[ystart:ystop, :, :]
    # scale down the image by factor 'scale'
    scaled_roi = roi
    if scale != 1:
        scaled_roi = cv2.resize(roi, (np.int(roi.shape[1] / scale), np.int(roi.shape[0] / scale)))

    hog_feature_cfg = feature_cfg.hog_feature_cfg
    hist_feature_cfg = feature_cfg.hist_feature_cfg
    color_feature_cfg = feature_cfg.color_feature_cfg

    hog_color_space = hog_feature_cfg["color_space"]
    img_for_hog = convert_color(scaled_roi, hog_color_space)

    if feature_cfg.use_hist_feature:
        if hist_feature_cfg["color_space"] == hog_color_space:
            img_for_hist = img_for_hog
        else:
            img_for_hist = convert_color(scaled_roi, hist_feature_cfg["color_space"])

    if feature_cfg.use_color_feature:
        if color_feature_cfg["color_space"] == hog_color_space:
            img_for_color = img_for_hog
        else:
            img_for_color = convert_color(scaled_roi, color_feature_cfg["color_space"])

    # Define blocks and steps
    pix_per_cell = hog_feature_cfg["pix_per_cell"]
    cells_per_block = hog_feature_cfg["cells_per_block"]
    orient = hog_feature_cfg["orient"]

    # the size of search window should be 64, the same size as the training image
    win_size = 64
    # the number of blocks per-window in one side
    nblocks_per_window = (win_size // pix_per_cell) - cells_per_block + 1
    # the stride for sliding window, described in cells other than pixels
    sliding_stride_in_cells = 2

    # image width and height in blocks
    nblocks_x = (img_for_hog.shape[1] // pix_per_cell) - cells_per_block + 1
    nblocks_y = (img_for_hog.shape[0] // pix_per_cell) - cells_per_block + 1

    nwindows_x = (nblocks_x - nblocks_per_window) // sliding_stride_in_cells
    nwindows_y = (nblocks_y - nblocks_per_window) // sliding_stride_in_cells

    # Compute individual channel HOG features for the entire image
    whole_img_hog = {}
    for c in hog_feature_cfg["channels"]:
        hog_img_channel = img_for_hog[:, :, c]
        whole_img_hog[c] = get_hog_features(hog_img_channel, orient, pix_per_cell, cells_per_block, feature_vec=False)


    detected_boxes = []
    for xb in range(nwindows_x):
        for yb in range(nwindows_y):
            # left top corner of current window in cells
            ypos = yb * sliding_stride_in_cells
            xpos = xb * sliding_stride_in_cells
            img_window_features = []
            if feature_cfg.use_hog_feature:
                # Extract HOG for this window
                for c in hog_feature_cfg["channels"]:
                    img_window_features.append(
                        whole_img_hog[c][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())

            # left top corner of current window in pixels
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            if feature_cfg.use_color_feature:
                # the image window used to extract color feature
                color_img_window = img_for_color[ytop:ytop + win_size, xleft:xleft + win_size]
                if color_img_window.shape[0] != 64 or color_img_window.shape[1] != 64:
                    color_img_window = cv2.resize(color_img_window, (64, 64))
                # Get color features
                color_feature = color_digest(color_img_window, color_feature_cfg["target_size"],
                                             color_feature_cfg["channels"])
                img_window_features.append(color_feature)

            if feature_cfg.use_hist_feature:
                # the image window used to extract histogram feature
                hist_img_window = img_for_hist[ytop:ytop + win_size, xleft:xleft + win_size]
                if hist_img_window.shape[0] != 64 or hist_img_window.shape[1] != 64:
                    hist_img_window = cv2.resize(hist_img_window, (64, 64))
                # get histogram features
                hist_feature = color_hist(hist_img_window, hist_feature_cfg["nbins"], hist_feature_cfg["channels"])
                img_window_features.append(hist_feature)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.concatenate(img_window_features).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = classifier.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                box_size = np.int(win_size * scale)
                detected_boxes.append(((xbox_left, ytop_draw + ystart),
                                       (xbox_left + box_size, ytop_draw + box_size + ystart)))

    return detected_boxes


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
