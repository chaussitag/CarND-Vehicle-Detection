#!/usr/bin/env python
# coding=utf-8

import cv2
import matplotlib.pyplot as plt

from feature_utils import convert_color, get_hog_features
from configure import default_feature_cfg

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


# test_image_path = "test_images/test1.jpg"
test_image_path_list = [
    "dataset/vehicles/GTI_MiddleClose/image0000.png",
    "dataset/non-vehicles/GTI/image8.png",
]

for image_path in test_image_path_list:
    hog_test(image_path)
