import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json
import pycocotools.mask as mask_tool

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
pred_path = "/mnt/truenas/upload/czh/simpledet_msrcnn/experiments/ms_r50v1_fpn_1x/coco_val2017_result.json"

coco = COCO(anno_path)
with open(pred_path, 'r') as f:
    coco_pred = json.load(f)

# tmp_pred = coco_pred[0]
# segm_list = [tmp_pred['segmentation'], coco_pred[1]['segmentation']]
# print(segm_list)
# print(mask_tool.iou([tmp_pred['segmentation']], segm_list, [False]))

def get_pred_data(coco_pred):
    pred_data = {}
    for each_pred in coco_pred:
        x1, y1, w, h = each_pred['bbox']
        each_pred['bbox'] = [x1, y1, x1+w, y1+h]
        img_id = each_pred['image_id']
        if img_id not in pred_data:
            pred_data[img_id] = [each_pred]
        else:
            pred_data[img_id].append(each_pred)
    return pred_data

def get_gt_data(img_id):
    anno_id = coco.getAnnIds(img_id)
    anno_data = coco.loadAnns(anno_id)
    bbox_list = []; cate_list = []; mask_list = []
    for anno in anno_data:
        x1, y1, w, h = anno['bbox']
        bbox_list.append([x1, y1, x1+w, y1+h])
        cate_list.append(anno['category_id'])
        mask = coco.annToRLE(anno)
        mask_list.append(mask)
    bbox_data = np.array(bbox_list)
    cate_data = np.array(cate_list)
    mask_data = np.array(mask_list)
    return bbox_data, cate_data, mask_data

pred_data = get_pred_data(coco_pred)
img_ids = list(pred_data.keys())

def compute_iou(pred_bbox, gt_bbox):
    x1 = gt_bbox[:, 0]
    y1 = gt_bbox[:, 1]
    x2 = gt_bbox[:, 2]
    y2 = gt_bbox[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    pred_area = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)

    xx1 = np.maximum(x1, pred_bbox[0])
    yy1 = np.maximum(y1, pred_bbox[1])
    xx2 = np.minimum(x2, pred_bbox[2])
    yy2 = np.minimum(y2, pred_bbox[3])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    
    inter = w * h
    ovr = inter / (areas + pred_area - inter)
    return ovr

def match_pred_gt(img_id):
    gt_bbox, gt_cate, gt_mask = get_gt_data(img_id)
    pred_list = pred_data[img_id]
    match_pair = []
    for each_pred in pred_list:
        cate_id = each_pred['category_id']
        bbox = each_pred['bbox']
        mask = each_pred['segmentation']
        score = each_pred['score']
        select_index = np.where(gt_cate == cate_id)[0]
        if select_index.shape[0] != 0:
            select_gt_bbox = gt_bbox[select_index]
            select_gt_mask = gt_mask[select_index]
            ovr = compute_iou(bbox, select_gt_bbox).reshape(-1,)
            mask_ovr = mask_tool.iou([mask], select_gt_mask.tolist(), [False])
            mask_ovr = np.array(mask_ovr).reshape(-1,)
            max_index = np.argmax(ovr); max_iou = np.max(ovr); mask_iou = mask_ovr[max_index]
            if max_iou > 0.1:
                match_gt_bbox = select_gt_mask[max_index]
                match_pair.append([each_pred, match_gt_bbox, max_iou, mask_iou])
    return match_pair

def process_img(img_id):
    match_pair = match_pred_gt(img_id)
    diff_data = []
    mask_score_data = []
    for each_pair in match_pair:
        pred_data, match_gt_bbox, iou_value, mask_iou = each_pair
        diff_data.append(mask_iou)
        mask_score_data.append(pred_data['mask_score'] * pred_data['score'])
    return np.array(diff_data).reshape(-1, 1), np.array(mask_score_data).reshape(-1, 1)

import concurrent.futures
diff_list = []; var_list = []
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    for index, res_data in enumerate(executor.map(process_img, img_ids)):
        diff_data, var_data = res_data
        diff_list.append(diff_data)
        var_list.append(var_data)
        if index % 1000 == 0:
            print("Processing %d/%d"%(index+1, len(img_ids)))
            
scatter_diff = np.vstack(diff_list)
scatter_var = np.vstack(var_list)
scatter_var = np.mean(scatter_var, axis=1).reshape(-1, 1)

fig = plt.figure(figsize=(10, 10))
var_list = []
diff_list = []
plt.scatter(scatter_var, scatter_diff, s=3, alpha=0.5)
# plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('mask score')
plt.ylabel('mask iou')
fig.savefig('scatter_res50_mask_iou.png', dpi=200)

