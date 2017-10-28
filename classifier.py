#!/usr/bin/env python
# coding=utf8

from train_data import load_dataset, load_feature_scaler
from feature_utils import sliding_window_search, draw_boxes
from configure import default_feature_cfg

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
import time

classifier_cache_path = osp.join(osp.dirname(osp.abspath(__file__)), "classifier.p")


def train_classifier():
    features, labels, _ = load_dataset()

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 10000)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.1, random_state=rand_state)

    t_start = time()
    tuned_parameters = [
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
        {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},
    ]
    clf = GridSearchCV(svm.SVC(), tuned_parameters)
    clf.fit(train_features, train_labels)
    print("tuning the svc model with following parameters takes %.2fs" % (time() - t_start,))
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
        pickle.dump(classifier)
    return classifier


def test_one_image(img_path):
    img = cv2.imread(img_path)
    classifier = get_classifier()
    feature_scaler = load_feature_scaler()
    img_for_draw = np.copy(img)
    ystart = 400
    ystop = 656
    scales = np.arange(1.5, 0.4, -0.25)
    for scale in scales:
        boxes = sliding_window_search(img, ystart, ystop, scale, classifier, feature_scaler, default_feature_cfg)
        draw_boxes(img_for_draw, boxes)

    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    axes.imshow(img_for_draw)
    plt.show()


if __name__ == "__main__":
    test_one_image("test_images/test6.jpg")