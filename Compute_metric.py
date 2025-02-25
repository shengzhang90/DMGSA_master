import numpy as np
import os
import SimpleITK as sitk
from utils.airway_metric import evaluation_branch_metrics,evaluation_branch_metrics_sizes
import csv
from tqdm import tqdm
# windows


def calculate_metric_normal(gt_path, pred_path, save_csv, save_path):
    fids = os.listdir(pred_path)
    Metrics = []
    idx = 1
    for fid in fids:
        print("assessing ", fid)
        gt = sitk.ReadImage(gt_path + fid)
        # gt = sitk.ReadImage(gt_path + fid.split('_LA')[0]+'.nii.gz')
        gt_array = sitk.GetArrayFromImage(gt)
        gt_voi = np.where(gt_array > 0)
        z, y, x = gt_array.shape
        z_min, z_max = min(gt_voi[0]), max(gt_voi[0])
        y_min, y_max = min(gt_voi[1]), max(gt_voi[1])
        x_min, x_max = min(gt_voi[2]), max(gt_voi[2])
        z_min, z_max = max(0,z_min-20), z_max
        y_min, y_max = max(0, y_min - 20), min(y, y_max + 20)
        x_min, x_max = max(0, x_min - 20), min(x, x_max + 20)
        try:
            pred = sitk.ReadImage(pred_path + fid.split('.nii.gz')[0] + '.nii.gz')
        except:
            continue
            pred = sitk.ReadImage(pred_path + fid.split('.nii.gz')[0] + '_0000.nii.gz')
        pred_array = sitk.GetArrayFromImage(pred)
        gt_ = gt_array[z_min:z_max,y_min:y_max,x_min:x_max]
        pred_ = pred_array[z_min:z_max,y_min:y_max,x_min:x_max]

        iou, dice, length_ratio, branch_ratio, pre, leakages, fnr,large_cp = evaluation_branch_metrics(fid, gt_, pred_,
                                                                                              False)
        # print(np.sum(pred_array))

        # print("|{}/{}|  ".format(idx, len(fids)), fid, iou, length_ratio, branch_ratio, pre)
        pred_final = np.zeros_like(pred_array)
        pred_final[z_min:z_max, y_min:y_max, x_min:x_max] = large_cp
        sitk.WriteImage(sitk.GetImageFromArray(pred_final.astype(np.uint8)), pred_path+fid)
        if save_csv:
            with open(save_path + "detail.csv", 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([fid, iou, dice, length_ratio, branch_ratio, pre, leakages, fnr])
                f.close()
        Metrics.append([iou, dice, length_ratio, branch_ratio, pre, leakages, fnr])
        # Metrics.append([dice, length_ratio, branch_ratio, pre])
        idx+=1
        # print(fid, result)
    print("************** Overall metric: **********\n", np.mean(Metrics, axis=0))
    print(np.std(Metrics, axis=0))


