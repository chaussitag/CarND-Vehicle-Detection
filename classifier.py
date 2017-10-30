#!/usr/bin/env python
# coding=utf8

from train_data import load_dataset, load_feature_scaler
from feature_utils import draw_boxes, single_window_features
from sliding_window import sliding_window_search
from configure import sliding_window_cfg, default_feature_cfg

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
import time
import glob

classifier_cache_path = osp.join(osp.dirname(osp.abspath(__file__)), "classifier.p")


def train_classifier():
    features, labels, _ = load_dataset()

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 10000)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.2, random_state=rand_state)

    # shuffle the training set
    train_features, train_labels = shuffle(train_features, train_labels)

    t_start = time.time()
    #######################################################################################
    # following grid search takes too long to train
    # tuned_parameters = [
    #     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    #     # {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},
    # ]
    # clf = GridSearchCV(svm.SVC(), tuned_parameters)
    #######################################################################################
    tuned_parameters = {'C': [1, 10]}
    clf = GridSearchCV(svm.LinearSVC(), tuned_parameters)
    clf.fit(train_features, train_labels)
    print("tuning the svc model with following parameters takes %.2fs" % (time.time() - t_start,))
    print(tuned_parameters)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    predicted_labels = clf.predict(test_features)
    print(classification_report(test_labels, predicted_labels))
    print()

    return clf


def get_classifier(force_retrain=False):
    if not force_retrain and osp.isfile(classifier_cache_path):
        with open(classifier_cache_path, "rb") as f:
            classifier = pickle.load(f)
            return classifier

    classifier = train_classifier()
    with open(classifier_cache_path, "wb") as f:
        pickle.dump(classifier, f)
    return classifier


def test_one_frame(img_path):
    img = cv2.imread(img_path)
    classifier = get_classifier()
    feature_scaler = load_feature_scaler()
    img_for_draw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sliding_cfg = sliding_window_cfg
    # sliding_cfg = {
    #     "scales": (1, ),
    #     "y_start_stop": ((400, 465), ),
    # }
    search_scales = sliding_cfg["scales"]
    y_start_stops = sliding_cfg["y_start_stop"]
    x_start_stops = sliding_cfg["x_start_stop"]
    detected_boxes = []
    for i, scale in enumerate(search_scales):
        y_start = y_start_stops[i][0]
        y_stop = y_start_stops[i][1]
        x_start = x_start_stops[i][0]
        x_stop = x_start_stops[i][1]
        boxes = sliding_window_search(img, y_start, y_stop, x_start, x_stop, scale,
                                      classifier, feature_scaler, default_feature_cfg)
        detected_boxes.extend(boxes)

    draw_boxes(img_for_draw, detected_boxes)
    file_name = img_path.split("/")[-1]
    save_path = "test_images/test_results/" + file_name
    cv2.imwrite(save_path, img_for_draw[:, :, [2, 1, 0]])
    # fig, axes = plt.subplots(1, 1)
    # axes.imshow(img_for_draw)
    # plt.show()


def test_one_sample(img_path):
    img = cv2.imread(img_path)
    feature = single_window_features(img, default_feature_cfg)
    feature = feature.reshape(1, -1)
    feature_scaler = load_feature_scaler()
    scaled_feature = feature_scaler.transform(feature)
    classifier = get_classifier()
    label = classifier.predict(scaled_feature)
    print("%s: %d" % (img_path, label))

if __name__ == "__main__":
    test_img_list = glob.glob("test_images/mytest*.jpg")
    for test_img_path in test_img_list:
        test_one_frame(test_img_path)
    #test_one_frame("/home/daiguozhou/girl.jpg")
    # test_one_sample("dataset/vehicles/GTI_Far/image0005.png")
    # test_one_sample("dataset/vehicles/GTI_Left/image0009.png")
    # test_one_sample("dataset/vehicles/GTI_MiddleClose/image0000.png")
    # test_one_sample("dataset/vehicles/GTI_Right/image0035.png")
    # test_one_sample("dataset/vehicles/KITTI_extracted/10.png")
    # print()
    # test_one_sample("dataset/non-vehicles/GTI/image1.png")
    # test_one_sample("dataset/non-vehicles/GTI/image4.png")
    # test_one_sample("dataset/non-vehicles/GTI/image6.png")
    # test_one_sample("dataset/non-vehicles/Extras/extra1.png")
    # test_one_sample("dataset/non-vehicles/Extras/extra6.png")
    #test_one_sample("debug/0_0.jpg")
