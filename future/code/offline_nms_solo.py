from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mutils

import mmcv
import numpy as np

data_dir = '/mnt/truenas/scratch/czh/mmdet/work_dirs/solo_fpn_4x_hr48_mst/'
solo_small_dir = data_dir + 'before_nms_900.pkl'
solo_large_dir = data_dir + 'before_nms_1100.pkl'

anno_dir = '/mnt/truenas/scratch/czh/data/future/annotations/val_new.json'
coco = COCO(anno_dir)
val_anno = mmcv.load(anno_dir)

solo_small = mmcv.load(solo_small_dir)
solo_large = mmcv.load(solo_large_dir)

def nms(dets, thresh=0.5):
    if len(dets) == 0:
        return dets
    dets = np.array(dets)
    rles = dets[:, 0]
    scores = dets[:, 1]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tmp_rle = [rles[order[1:]]] if type(rles[order[1:]]) == dict else rles[order[1:]]
        ovr = mutils.iou([rles[i]], rles[order[1:]].tolist(), [False])
        if len(ovr) == 0:
            break
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]

def matrix_nms(dets, kernel='gaussian', sigma=.5):
    N = len(dets)
    if N == 0:
        return dets
    dets = np.array(dets)
    rles = dets[:, 0]
    scores = dets[:, 1]
    order = scores.argsort()[::-1]
    # sort in decending order
    sorted_scores = scores[order]
    sorted_rles = rles[order]
    ious = mutils.iou(rles.tolist(), rles.tolist(), [False])
    ious = np.triu(ious, k=1)
    # np.set_printoptions(precision=4, suppress=True)
    # print(ious)
    # exit()
    
    ious_cmax = ious.max(0)
    ious_cmax = np.tile(ious_cmax, reps=(N, 1)).T
    if kernel == 'gaussian':
        decay = np.exp(-(ious ** 2 - ious_cmax ** 2) / sigma)
    else: # linear
        decay = (1 - ious) / (1 - ious_cmax)
    # decay factor: N
    decay = decay.min(axis=0)
    sorted_scores *= decay
    dets = np.concatenate([sorted_rles.reshape(-1, 1), sorted_scores.reshape(-1, 1)], axis=-1)
    valid_ind = np.where(sorted_scores >= 0.05)[0]
    return dets[valid_ind]


def nms_per_img(dets):
    num_classes = len(dets)
    ensemble_dets = []
    for cid in range(num_classes):
        dets_per_class = dets[cid]
        dets_per_class = nms(dets_per_class)
        ensemble_dets.append(dets_per_class)
    return ensemble_dets

def group_results_img(det_list):
    num_ensembles = len(det_list)
    num_images = len(det_list[0])
    num_classes = len(det_list[0][0])
    group_dets = []
    for img_id in range(num_images):
        group_dets_per_img = [[] for ii in range(num_classes)]
        for model_id in range(num_ensembles):
            model_det_per_img = det_list[model_id][img_id]
            # NOTE: process model_det_per_img here(multiply coeficient)
            for cid in range(num_classes):
                group_dets_per_img[cid] += model_det_per_img[cid]
        group_dets.append(group_dets_per_img)
    return group_dets   

group_dets = group_results_img([solo_large])
new_dets = []
import concurrent.futures
with concurrent.futures.ProcessPoolExecutor(max_workers=45) as executor:
    for index, res_data in enumerate(executor.map(nms_per_img, group_dets)):
        new_dets.append(res_data)
# for index, group_det_per_img in enumerate(group_dets):
#     new_dets.append(nms_per_img(group_det_per_img))

from parse_result import parse_pred_2_json
tmp_res = parse_pred_2_json(new_dets, val_anno)
coco_dt = coco.loadRes(tmp_res['annotations'])
coco_eval = COCOeval(coco, coco_dt)
ann_type = 'segm'
coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 113 ** 2], [113 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
coco_eval.params.useSegm = (ann_type == 'segm')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

