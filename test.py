#!/usr/bin/env python
# coding=utf-8

import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from feature_utils import convert_color, get_hog_features
from configure import default_feature_cfg, sliding_window_cfg

hog_feature_cfg = default_feature_cfg.hog_feature_cfg
color_feature_cfg = default_feature_cfg.color_feature_cfg
hist_feature_cfg = default_feature_cfg.hist_feature_cfg


def hog_test(img_path):
    bgr_image = cv2.imread(img_path)

    orient = hog_feature_cfg["orient"]
    pix_per_cell = hog_feature_cfg["pix_per_cell"]
    cells_per_block = hog_feature_cfg["cells_per_block"]
    img_for_hog = convert_color(bgr_image, hog_feature_cfg["color_space"])
    hog_imgs = []
    for c in hog_feature_cfg["channels"]:
        _, hog_img = get_hog_features(img_for_hog[:, :, c], orient, pix_per_cell, cells_per_block,
                                      vis=True, feature_vec=True)
        hog_imgs.append(hog_img)

    n = len(hog_imgs)
    fig, axes = plt.subplots(1, n + 1)
    axes[0].imshow(bgr_image[:, :, [2, 1, 0]])
    axes[0].set_title("original")
    for c in range(n):
        axes[c+1].set_title("ch-%d hog" % c)
        axes[c + 1].imshow(hog_imgs[c], cmap="gray")
    plt.tight_layout()
    plt.show()


def sliding_config_test(img_path):
    img_name = img_path.split("/")[-1]
    bgr_image = cv2.imread(img_path)
    scales = sliding_window_cfg["scales"]
    x_start_stops = sliding_window_cfg["x_start_stop"]
    y_start_stops = sliding_window_cfg["y_start_stop"]
    scale_indices = range(len(scales))
    #scale_indices = [2]
    for i in scale_indices:
        scale = scales[i]
        fig, axes = plt.subplots(1, 1)
        axes.imshow(bgr_image[:, :, [2, 1, 0]])
        axes.set_title("%s: scale %.2f" % (img_name, scale))
        xlr = x_start_stops[i]
        ytb = y_start_stops[i]
        w = xlr[1] - xlr[0]
        h = ytb[1] - ytb[0]
        axes.add_patch(Rectangle((xlr[0], ytb[0]), w, h, linewidth=1, edgecolor='r', facecolor='none'))
        axes.add_patch(Rectangle((xlr[0], ytb[0]), int(64*scale), int(64*scale), linewidth=1, edgecolor='g', facecolor='none'))
        plt.show()


# # test_image_path = "test_images/test1.jpg"
# test_image_path_list = [
#     "dataset/vehicles/GTI_MiddleClose/image0000.png",
#     "dataset/non-vehicles/GTI/image8.png",
# ]
#
# for image_path in test_image_path_list:
#     hog_test(image_path)

test_img_list = glob.glob("test_images/white_loss*.jpg")
for test_img_path in test_img_list:
    sliding_config_test(test_img_path)