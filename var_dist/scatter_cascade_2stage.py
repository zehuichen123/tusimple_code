import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
pred_path = '/mnt/truenas/upload/czh/simpledet_xyxy/experiments/cascade_r50v1_fpn_1x_2stage/coco_val2017_result.json'
coco = COCO(anno_path)
with open(pred_path, 'r') as f:
    coco_pred = json.load(f)

def get_pred_data(coco_pred):
    pred_data = {}
    for each_pred in coco_pred:
        x1, y1, w, h = each_pred['bbox']
        xx1, yy1, ww, hh = each_pred['bbox_2nd']
        each_pred['bbox'] = [x1, y1, x1+w, y1+h]
        each_pred['bbox_2nd'] = [xx1, yy1, xx1+ww, yy1+hh]
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
        proposal = each_pred['bbox']
        delta = each_pred['delta']
        score = each_pred['score']
        delta_2nd = each_pred['delta_2nd']
        select_index = np.where(gt_cate == cate_id)[0]
        if select_index.shape[0] != 0:
            select_gt_bbox = gt_bbox[select_index]
            ovr = compute_iou(proposal, select_gt_bbox)
            max_index = np.argmax(ovr); max_iou = np.max(ovr)
            if max_iou > 0.5:
                match_gt_bbox = select_gt_bbox[max_index]
                match_pair.append([each_pred, match_gt_bbox, max_iou])
    return match_pair

def process_img(img_id):
    match_pair = match_pred_gt(img_id)
    diff_data = [[] for i in range(4)]
    delta_data = [[] for i in range(4)]
    diff_2nd_data = [[] for i in range(4)]
    delta_2nd_data = [[] for i in range(4)]
    iou_data = []
    for each_pair in match_pair:
        pred_data, match_gt_bbox, iou_score = each_pair
        bbox_2nd = pred_data['bbox_2nd']
        delta_2nd = pred_data['delta_2nd']
        delta = pred_data['delta']
        bbox = pred_data['bbox']
        score = pred_data['score']
        iou_data.append(iou_score)
        w_2nd = bbox_2nd[2] - bbox_2nd[0] + 1
        h_2nd = bbox_2nd[3] - bbox_2nd[1] + 1
        width = bbox[2] - bbox[0] + 1
        height = bbox[3] - bbox[1] + 1
        for ii in range(4):
            if ii % 2 == 0:
                diff_data[ii].append((match_gt_bbox[ii] - bbox[ii]) / width)
                diff_2nd_data[ii].append((match_gt_bbox[ii] - bbox_2nd[ii]) / width)
            else:
                diff_data[ii].append((match_gt_bbox[ii] - bbox[ii]) / height)
                diff_2nd_data[ii].append((match_gt_bbox[ii] - bbox_2nd[ii]) / height)
            delta_data[ii].append(delta[ii])
            delta_2nd_data[ii].append(delta_2nd[ii])
    return np.array(diff_data).transpose(), np.array(delta_data).transpose(), np.array(diff_2nd_data).transpose(), np.array(delta_2nd_data).transpose()

import concurrent.futures
diff_list = []; delta_list = []; diff_2nd_list = []; delta_2nd_list = []
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    for index, res_data in enumerate(executor.map(process_img, img_ids)):
        diff_data, delta_data, diff_2nd_data, delta_2nd_data = res_data
        diff_list.append(diff_data)
        delta_list.append(delta_data)
        diff_2nd_list.append(diff_2nd_data)
        delta_2nd_list.append(delta_2nd_data)
        if index % 1000 == 0:
            print("Processing %d/%d"%(index+1, len(img_ids)))
            
scatter_diff = np.vstack(diff_list)
# scatter_delta = abs(np.vstack(delta_list)) * 0.1
scatter_diff_2nd = np.vstack(diff_2nd_list)
# scatter_delta_2nd = abs(np.vstack(delta_2nd_list)) * 0.1

# scatter_diff = abs(scatter_diff - scatter_delta)
# scatter_delta = abs(scatter_diff_2nd - scatter_delta_2nd)
# np.save('data/diff.npy', scatter_diff)
# np.save('data/var.npy', scatter_var)
# np.save('data/iou.npy', scatter_iou)
scatter_delta = scatter_diff_2nd

print(scatter_diff.shape)
# print(scatter_delta.shape)

title_list = ['x1', 'y1', 'x2', 'y2']
fig = plt.figure(figsize=(10, 10))
for ii in range(4):
    fig.add_subplot("22"+str(ii+1))
    var_list = []
    diff_list = []
    coef = np.corrcoef(scatter_delta[:, ii], scatter_diff[:, ii])[0, 1]
    plt.scatter(scatter_delta[:, ii], scatter_diff[:, ii], s=3, alpha=0.05)
    plt.xlabel('2nd diff')
    plt.ylabel('diff')
    plt.plot([0, 1], [0, 1], c='black')
    plt.plot([0, -1], [0, -1], c='black')
    plt.grid(True)
    plt.title(title_list[ii] + ' ' +str(coef))
    # plt.xlim(-0.01, 0.9)
    # plt.ylim(-0.01, 0.9)
fig.savefig('img/delta_vs_diff', dpi=200)


# title_list = ['x1', 'y1', 'x2', 'y2']
# fig = plt.figure(figsize=(10, 10))
# for ii in range(4):
#     fig.add_subplot("22"+str(ii+1))
#     plt.scatter(scatter_var[:, ii], scatter_diff[:, ii], s=3, alpha=0.5)
#     plt.xlabel('var')
#     plt.ylabel('diff')
#     plt.plot([0, 2], [0, 2], c='red')
#     plt.title(title_list[ii])
#     plt.xlim(-0.1, 6)
#     plt.ylim(-0.1, 3)
# fig.savefig('res152_proposal.png', dpi=200)


