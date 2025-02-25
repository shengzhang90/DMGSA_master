import torch
import numpy as np
import os
import nibabel
import skimage.measure as measure
from skimage.morphology import skeletonize_3d
from utils.tree_parse import get_parsing
import math

EPSILON = 1e-32

def compute_binary_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred) + EPSILON
    union = np.sum(y_true) + np.sum(y_pred) - intersection + EPSILON
    iou = intersection / union
    return iou

def compute_binary_dice(y_true, y_pred):
    intersection = 2 * y_true * y_pred
    union = y_true + y_pred
    dice = (np.sum(intersection) + EPSILON) / (np.sum(union) + EPSILON)
    return dice

def evaluation_branch_metrics(fid,label, pred,refine=False):
    """
    :return: length_ratio, branch_ratio
    """
    # compute tree sparsing
    parsing_gt = get_parsing(label, refine)
    # this aims to find the largest component to locate the airway prediction
    cd, num = measure.label(pred, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    # calculate iou
    iou = compute_binary_iou(label, large_cd)
    # make sure the extracted largest component is correct
    jj=-1
    while iou < 0.1:
        print(fid," failed need post-processing")
        jj -= 1
        large_cd = (cd == (volume_sort[jj] + 1)).astype(np.uint8)
        iou = compute_binary_iou(label, large_cd)
        if jj == -5:
            break
    skeleton = skeletonize_3d(label)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    dice = compute_binary_dice(label, large_cd)
    DLR = (large_cd * skeleton).sum() / skeleton.sum()
    pre = (large_cd * label).sum() / large_cd.sum()
    leakages = ((large_cd - label)==1).sum() / label.sum()
    FNR = ((label - large_cd)==1).sum() / label.sum()

    num_branch = parsing_gt.max()
    detected_num = 0
    for j in range(num_branch):
        branch_label = ((parsing_gt == (j + 1)).astype(np.uint8)) * skeleton
        if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
            detected_num += 1
    DBR = detected_num / num_branch
    return iou,dice, DLR, DBR, pre, leakages, FNR, large_cd
