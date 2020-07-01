from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mutils

import mmcv
import numpy as np

data_dir = '/mnt/truenas/scratch/czh/mmdet2.0/work_dirs/htc_res2_fpn_4x_gcb_dcn_mst_largemaskalign_mscore_fp16/'
htc_small_dir = data_dir + 'before_nms_800.pkl'
htc_large_dir = data_dir + 'before_nms_1000.pkl'

anno_dir = '/mnt/truenas/scratch/czh/data/future/annotations/val_new.json'
coco = COCO(anno_dir)
val_anno = mmcv.load(anno_dir)

htc_small = mmcv.load(htc_small_dir)
htc_large = mmcv.load(htc_large_dir)
print("Load Done")

def mask_nms(dets, thresh=0.6):
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
    # return keep

def box_nms(dets, thresh=0.5):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


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
    return dets

def nms_per_img(dets):
    ensemble_dets = []
    mask_info = dets[1]; bbox_data = dets[0]
    num_classes = len(bbox_data)
    if len(mask_info) == 2:
        mask_data = mask_info[0]; score_data = mask_info[1]
    else:
        print(("Mask Scoring Not Used"))
        raise Exception
        mask_data = mask_info; score_data = [[] for ii in range(num_classes)]
    for cid in range(num_classes):
        bboxes = np.array(bbox_data[cid])
        # if bboxes.shape[0] == 0:
        #     continue
        masks = np.array(mask_data[cid]).reshape(-1, 1)
        scores = 1 if len(score_data[cid]) == 0 else np.array(score_data[cid]).reshape(-1,)
        keep = box_nms(bboxes)
        final_scores = bboxes[:, -1] * scores
        dets_per_class = np.concatenate([masks, final_scores.reshape(-1, 1)], axis=-1)
        valid_ind = dets_per_class[:, 1]
        ensemble_dets.append(dets_per_class[keep])
    assert len(ensemble_dets) == num_classes
    return ensemble_dets

# def nms_per_img(dets):
#     num_classes = len(dets[0])
#     ensemble_dets = []
#     mask_info = dets[1]; bbox_data = dets[0]
#     if len(mask_info) == 2:
#         mask_data = mask_info[0]; score_data = mask_info[1]
#     else:
#         mask_data = mask_info; score_data = [[] for ii in range(num_class)]
#     for cid in range(num_classes):
#         bboxes = np.array(bbox_data[cid])
#         masks = np.array(mask_data[cid]).reshape(-1, 1)
#         scores = 1 if len(score_data[cid]) == 0 else np.array(score_data[cid])
#         final_scores = bboxes[:, -1] * scores
#         dets_per_class = np.concatenate([masks, final_scores.reshape(-1, 1)], axis=-1)
#         dets_per_class = mask_nms(dets_per_class)
#         ensemble_dets.append(dets_per_class)
#     return ensemble_dets

def scale_aware_weighted(dets, small_scale_test=True):
    # default set 256 as split
    split_size = 256
    decay_ratio = 0
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    small_ind = areas <= split_size ** 2
    large_ind = areas > split_size ** 2
    if small_scale_test:
        dets[small_ind, 4] *= decay_ratio
    else:
        dets[large_ind, 4] *= decay_ratio
    return dets

def group_results_img(det_list, scale_list=[True, False]):
    num_ensembles = len(det_list)
    num_images = len(det_list[0])
    num_classes = len(det_list[0][0][0])
    group_dets = []
    for img_id in range(num_images):
        group_bbox_per_img = [[] for ii in range(num_classes)]
        group_segm_per_img = [[[] for ii in range(num_classes)], [[] for ii in range(num_classes)]]
        for model_id in range(num_ensembles):
            bbox_data, mask_info = det_list[model_id][img_id]
            # NOTE: process model_det_per_img here(multiply coeficient)
            for cid in range(num_classes):
                bbox_data[cid] = scale_aware_weighted(bbox_data[cid], scale_list[model_id])
                group_bbox_per_img[cid].append(bbox_data[cid])
                group_segm_per_img[0][cid] += mask_info[0][cid]
                group_segm_per_img[1][cid].append(mask_info[1][cid].reshape(-1, 1))
        for cid in range(num_classes):
            group_bbox_per_img[cid] = np.vstack(group_bbox_per_img[cid])
            group_segm_per_img[1][cid] = np.vstack(group_segm_per_img[1][cid])
            assert group_bbox_per_img[cid].shape[0] == group_segm_per_img[1][cid].shape[0], (group_bbox_per_img[cid].shape, group_segm_per_img[1][cid].shape)
        group_dets.append([group_bbox_per_img, group_segm_per_img])
    return group_dets   

group_dets = group_results_img([htc_small, htc_large])
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

