import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
pred_path = "/mnt/truenas/upload/czh/simpledet_baseline2/experiments/faster_r152v1b_fpn_1x_xyxy_paper_l1/coco_val2017_result_proposal.json"
# pred_path = "/mnt/truenas/upload/czh/simpledet_baseline2/experiments/faster_r50v1b_fpn_1x_xyxy_paper_l1/coco_val2017_result_proposal.json"

coco = COCO(anno_path)
with open(pred_path, 'r') as f:
    coco_pred = json.load(f)

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
    bbox_list = []; cate_list = []
    for anno in anno_data:
        x1, y1, w, h = anno['bbox']
        bbox_list.append([x1, y1, x1+w, y1+h])
        cate_list.append(anno['category_id'])
    bbox_data = np.array(bbox_list)
    cate_data = np.array(cate_list)
    return bbox_data, cate_data

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

pred_data = get_pred_data(coco_pred)
img_ids = list(pred_data.keys())

def match_pred_gt(img_id):
    gt_bbox, gt_cate = get_gt_data(img_id)
    pred_list = pred_data[img_id]
    match_pair = []
    for each_pred in pred_list:
        cate_id = each_pred['category_id']
        bbox = each_pred['bbox']
        select_index = np.where(gt_cate == cate_id)[0]
        if select_index.shape[0] != 0:
            select_gt_bbox = gt_bbox[select_index]
            ovr = compute_iou(bbox, select_gt_bbox)
            max_index = np.argmax(ovr); max_iou = np.max(ovr)
            if max_iou > 0.1:
                match_gt_bbox = select_gt_bbox[max_index]
                match_pair.append([each_pred, match_gt_bbox, max_iou])
    return match_pair

def process_img(img_id):
    match_pair = match_pred_gt(img_id)
    diff_data = []
    var_data = [[] for i in range(4)]
    for each_pair in match_pair:
        pred_data, match_gt_bbox, iou_value = each_pair
        var = pred_data['var']
        diff_data.append(iou_value)
        for ii in range(4):
            var_data[ii].append(var[ii])
    return np.array(diff_data).reshape(-1, 1), np.array(var_data).transpose()

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

def compute_mean_var(thres, var_data, diff_data):
    select_index = np.where((diff_data > thres) & (diff_data <= thres + 0.1))[0]
    mean_var = np.mean(var_data[select_index])
    mean_diff = np.mean(diff_data[select_index])
    return mean_var, mean_diff

fig = plt.figure(figsize=(10, 10))
var_list = []
diff_list = []
for jj in np.arange(0.1, 1, 0.1):
    mean_var, mean_diff = compute_mean_var(jj, scatter_var, scatter_diff)
    var_list.append(mean_var)
    diff_list.append(mean_diff)
plt.plot(var_list, diff_list, marker='.', markersize=15, linewidth=5)
plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('var')
plt.ylabel('iou')
fig.savefig('res152_iou_proposal.png', dpi=200)

