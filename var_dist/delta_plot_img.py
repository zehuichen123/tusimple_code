import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json
import cv2

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
pred_path = '/mnt/truenas/upload/czh/simpledet_xyxy/experiments/cascade_r50v1_fpn_1x_2stage/coco_val2017_result.json'
# pred_path = '/mnt/truenas/upload/czh/simpledet_baseline4/experiments/faster_r50_1x_paper_l1_varhead_2fc_gs1_woexp/coco_val2017_result.json'
# pred_path = '/mnt/truenas/upload/czh/simpledet_baseline5/experiments/faster_r50_1x_paper_l1_varhead_2fc_gs1_abstarget/coco_val2017_result.json'
coco = COCO(anno_path)

with open(pred_path, 'r') as f:
    coco_pred = json.load(f)
cats = coco.loadCats(coco.getCatIds())
new_cats = {}
for each_cat in cats:
    new_cats[each_cat['id']] = each_cat['name']


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
img_ids = list(pred_data.keys())[:200]

def match_pred_gt(img_id):
    gt_bbox, gt_cate = get_gt_data(img_id)
    pred_list = pred_data[img_id]
    match_pair = []
    for each_pred in pred_list:
        cate_id = each_pred['category_id']
        proposal = each_pred['proposal']
        delta = each_pred['delta']
        score = each_pred['score']
        select_index = np.where(gt_cate == cate_id)[0]
        if select_index.shape[0] != 0:
            select_gt_bbox = gt_bbox[select_index]
            ovr = compute_iou(proposal, select_gt_bbox)
            max_index = np.argmax(ovr); max_iou = np.max(ovr)
            if max_iou > 0.5:
                match_gt_bbox = select_gt_bbox[max_index]
                gt_list = abs(match_gt_bbox[:4] - each_pred['proposal'])
                gt_list[0::2] /= (proposal[2] - proposal[0] + 1)
                gt_list[1::2] /= (proposal[3] - proposal[1] + 1)
                delta = [abs(0.1 * x) for x in delta]
                diff_delta = max(abs(gt_list - delta))
                # print(diff_delta)
                for ii in range(4):
                    if delta[ii] < 0.05 and gt_list[ii] > 0.2:
                        each_pred['gt_bbox'] = match_gt_bbox.tolist()
                        match_pair.append(each_pred)
                        break
                
    return match_pair

def process_img(img_id):
    match_pair = match_pred_gt(img_id)
    return match_pair

import concurrent.futures
match_list = []
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    for index, res_data in enumerate(executor.map(process_img, img_ids)):
        match_pair = res_data
        match_list.append(match_pair)
        if index % 1000 == 0:
            print("Processing %d/%d"%(index+1, len(img_ids)))

color_list = [[0, 153, 51], [0, 153, 255], [255, 255, 0], [204, 102, 255], [255, 0, 0], [204, 0, 255], [102, 255, 255], [255, 102, 102]]
def draw_after_bbox(img, bbox_score_data, index, show_box, show_score, show_class, cid, is_gt=0):
    global color_list, new_cats
    index %= 6
    index += 2

    cat_name = new_cats[cid]
    test = list(map(lambda x: int(x), bbox_score_data[:4]))
    if show_box:
        if is_gt == 0:
            img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[index], 1)
        if is_gt == 1:
            img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[0], 2)
        if is_gt == 2:
            img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[1], 2)
    if show_score or show_class:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ""
        if show_score:
            text += str(bbox_score_data[4])[:4]
        if show_class:
            text += cat_name
        rec_width = len(text) * 6
        if is_gt == 0:
            img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
            cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    return img

data_path = '/mnt/truenas/upload/czh/data/coco/coco/images/val2017/'

def draw_img(each_img_data):
    if len(each_img_data) == 0:
        return None
    img_id = each_img_data[0]['image_id']
    img_path = data_path + '%012d.jpg'%img_id
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for index, each_pred in enumerate(each_img_data):
        img = draw_after_bbox(img, each_pred['gt_bbox'][:4] + [-1.], index, True, False, False, each_pred['category_id'], 1)
        img = draw_after_bbox(img, each_pred['bbox'] + [each_pred['score']], index, True, True, True, each_pred['category_id'])
        img = draw_after_bbox(img, each_pred['proposal'] + [-1.], index, True, False, False, each_pred['category_id'], 2)
    plt.imsave('coco_img/%012d.jpg'%img_id, img)
    return True

for each_data in match_list:
    draw_img(each_data)

    






