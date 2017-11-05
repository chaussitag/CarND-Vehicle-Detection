#!/usr/bin/env python
# coding=utf-8

from classifier import get_classifier
from train_data import load_feature_scaler
from sliding_window import sliding_window_search
from configure import sliding_window_cfg, default_feature_cfg

import argparse
import cv2
import numpy as np
from scipy.ndimage.measurements import label
import os.path as osp
from moviepy.editor import VideoFileClip

import matplotlib.pyplot as plt
from feature_utils import draw_boxes


class VehicleDetector(object):
    def __init__(self):
        self._heat_maps = []
        self._heat_threshold = 24
        self._max_kept = 8
        self._classifier = get_classifier()
        self._feature_scaler = load_feature_scaler()
        self._frame_id = 0
        self._debug = False

    def append_heat_map(self, heat_map):
        if len(self._heat_maps) >= self._max_kept:
            self._heat_maps.pop(0)
        self._heat_maps.append(heat_map)

    def detect(self, rgb_img):
        self._frame_id += 1
        bgr_img = rgb_img[:, :, [2, 1, 0]]
        search_scales = sliding_window_cfg["scales"]
        y_start_stops = sliding_window_cfg["y_start_stop"]
        x_start_stops = sliding_window_cfg["x_start_stop"]
        # box_colors = sliding_window_cfg["box_color"]
        detected_boxes = []
        for i, scale in enumerate(search_scales):
            y_start = y_start_stops[i][0]
            y_stop = y_start_stops[i][1]
            x_start = x_start_stops[i][0]
            x_stop = x_start_stops[i][1]
            boxes = sliding_window_search(bgr_img, y_start, y_stop, x_start, x_stop, scale,
                                          self._classifier, self._feature_scaler, default_feature_cfg, 0.85)
            if len(boxes) > 0:
                detected_boxes.extend(boxes)

        # create heat map
        if len(detected_boxes) > 0:
            heat_map = np.zeros(rgb_img.shape[0:2])
            # Iterate through list of bboxes
            for box in detected_boxes:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
            self.append_heat_map(heat_map)
            #############################################
            # for debugging only
            if self._debug:
                fig, axes = plt.subplots(1, 2)
                draw_img = np.copy(rgb_img)
                draw_boxes(draw_img, detected_boxes)
                axes[0].set_title("frame %d" % self._frame_id)
                axes[0].imshow(draw_img)
                color_heat_map = np.dstack((heat_map, np.zeros_like(heat_map), np.zeros_like(heat_map)))
                color_heat_map = 255 / np.max(heat_map) * color_heat_map
                axes[1].set_title("heat map")
                axes[1].imshow(color_heat_map)
                plt.show()
            #############################################

        if len(self._heat_maps) >= self._max_kept:
            sumed_heat_map = np.sum(self._heat_maps, axis=0)
            sumed_heat_map[sumed_heat_map <= self._heat_threshold] = 0
            labels = label(sumed_heat_map)
            self._draw_labeled_bboxes(rgb_img, labels)
            #############################################
            # for debugging only
            if self._debug:
                fig, axes = plt.subplots(1, 1)
                axes.set_title("labels")
                axes.imshow(labels[0], cmap="gray")
                plt.show()
                fig1, axes1 = plt.subplots(1, 1)
                axes1.imshow(rgb_img)
                plt.show()
            #############################################

        return rgb_img

    @staticmethod
    def _draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            w = bbox[1][0] - bbox[0][0] + 1
            h = bbox[1][1] - bbox[0][1] + 1
            # ignore portrait or very small box
            if (0.75 * h) > w or h <= 30 or w <= 30:
                continue
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img


def get_frame_process_func():
    detector = VehicleDetector()

    def process_frame(frame):
        return detector.detect(frame)

    return process_frame


if __name__ == "__main__":
    dir_of_this_file = osp.dirname(osp.abspath(__file__))
    # default_video_file = osp.join(dir_of_this_file, "white_failed.mp4")
    default_video_file = osp.join(dir_of_this_file, "project_video.mp4")

    parser = argparse.ArgumentParser("vehicle detector")
    parser.add_argument("--input", "-i", help="path to the input video, default to project_video.mp4",
                        default=default_video_file)
    parser.add_argument("--output", "-o",
                        help="path to the output video, append '_output' to input name if not specified")

    args = parser.parse_args()

    if not osp.isfile(args.input):
        parser.error("the input %s is not a valid file" % args.input)

    if args.output is None:
        input_name = args.input.split("/")[-1]
        name_splits = input_name.split(".")
        args.output = osp.join(dir_of_this_file, name_splits[0] + "_output." + name_splits[-1])
    print("output is %s" % args.output)

    video_clip = VideoFileClip(args.input)
    process_frame_func = get_frame_process_func()
    white_clip = video_clip.fl_image(process_frame_func)  # NOTE: this function expects color images!!
    white_clip.write_videofile(args.output, audio=False)
