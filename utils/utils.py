import os
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
import numpy as np
from skimage import measure
from scipy import ndimage
from pylab import figure, subplot, imshow, show


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value/normalize for key, value in self.results.items()}


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.
    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def adjust_window(image, window_center=-300, window_width=1800):
    win_min = window_center - window_width // 2
    win_max = window_center + window_width // 2
    image = 255.0 * (image - win_min) / (win_max - win_min)
    image[image>255] = 255
    image[image<0] = 0
    return image

def zcore_normalization(images, p995,p005, mean=None, std=None):
    if mean==None or std==None:
        raise Exception("compute the mean and std of training set first!")
    # images = np.clip(images, p005, p995)
    images = (images - mean)/std
    return images

def extract_patches(X, patch_size, dim):
    l = X.shape[dim]
    patches = []
    for i in range(0, l, patch_size):
        start_idex = min(i, l-patch_size)
        end_idex = min(l, i+patch_size)
        if dim ==0:
            patch = X[start_idex:end_idex, :, :]
        elif dim == 1:
            patch = X[:, start_idex:end_idex, :]
        elif dim == 2:
            patch = X[:,:, start_idex:end_idex]
        patches.append(patch)
    return patches


def get_largest_target(roi):
    class_lists = np.unique(roi).tolist()
    area_list = [np.sum(roi==idex) for idex in class_lists]
    return class_lists[np.argmax(area_list)]
