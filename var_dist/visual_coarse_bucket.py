import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json
import cv2

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
pred_path = '/mnt/truenas/upload/czh/simpledet_tmp/experiments/sabl_r50v1_fpn_1x_conv/coco_val2017_result.json'
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
        each_pred['bucket_logit'] = np.array(each_pred['bucket_logit'])
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
        bbox = each_pred['coarse']
        score = each_pred['score']
        area = (bbox[3] - bbox[1] + 1) * (bbox[2] - bbox[0] + 1)
        select_index = np.where(gt_cate == cate_id)[0]
        if select_index.shape[0] != 0:
            select_gt_bbox = gt_bbox[select_index]
            ovr = compute_iou(bbox, select_gt_bbox)
            max_index = np.argmax(ovr); max_iou = np.max(ovr)
            if max_iou > 0.5 and score > 0.5 and area > 10000:
                match_gt_bbox = select_gt_bbox[max_index]
                match_pair.append([each_pred, match_gt_bbox])
    return match_pair

def compute_gt_bucket(bbox, gt_bbox):
    width = bbox[2] - bbox[0] + 1
    height = bbox[3] - bbox[1] + 1
    target_x1 = gt_bbox[0] - bbox[0]
    target_y1 = gt_bbox[1] - bbox[1]
    target_x2 = gt_bbox[2] - bbox[2]
    target_y2 = gt_bbox[3] - bbox[3]
    target_list = [target_x1, target_y1, target_x2, target_y2]
    label = []
    for target in target_list:
        floor_target = np.floor(target)
        floor_target = max(0, floor_target)
        floor_target = min(6, floor_target)
        label.append(floor_target)
    return label

def compute_pred_bucket(pred_logits):
    pred_logits = pred_logits.reshape(4, 7)
    pred_bucket = np.argmax(pred_logits, axis=1).astype(np.int32)
    return pred_bucket

def process_img(img_id):
    match_pair = match_pred_gt(img_id)
    pair_list = []
    for each_pair in match_pair:
        pred_data, match_gt_bbox = each_pair
        proposal = pred_data['proposal']
        bbox = pred_data['coarse']
        gt_bucket = compute_gt_bucket(bbox, match_gt_bbox) # (4, )
        pred_bucket = compute_pred_bucket(pred_data['bucket_logit'])
        if max(abs(gt_bucket - pred_bucket)) > 0:
            pred_data['bucket_pred'] = pred_bucket
            pred_data['gt_bbox'] = match_gt_bbox
            pair_list.append(pred_data)
    return pair_list
    

import concurrent.futures
match_list = []
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    for index, res_data in enumerate(executor.map(process_img, img_ids)):
        match_pair = res_data
        match_list.append(match_pair)
        if index % 1000 == 0:
            print("Processing %d/%d"%(index+1, len(img_ids)))

color_list = [[0, 153, 51], [0, 153, 255], [255, 255, 0], [204, 102, 255], [255, 0, 0], [204, 0, 255], [102, 255, 255], [255, 102, 102]]

def draw_after_bbox(img, bbox, score, index, cid, pred_logits=None, pred_bucket=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # show bbox
    test = list(map(lambda x: int(x), bbox))
    img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[index], 2)
    # show score and class
    if index == 0:
        text = ""
        cat_name = new_cats[cid]
        text += str(score)[:4]
        text += cat_name
        rec_width = len(text) * 6
        img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
        cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    # show bucket
    if index == 1:
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        bucket_w = width / 14; bucket_h = height / 14
        bucket_bin_w = [int(bbox[0] + ii * bucket_w) for ii in range(1, 14)]
        bucket_bin_h = [int(bbox[1] + ii * bucket_h) for ii in range(1, 14)]
        for ii in range(13):
            img = cv2.line(img, (bucket_bin_w[ii], test[1] - 1), (bucket_bin_w[ii], test[1] + 1), (255, 255, 255), 3)
            img = cv2.line(img, (test[0] - 1, bucket_bin_h[ii]), (test[0] + 1, bucket_bin_h[ii]), (255, 255, 255), 3)
            img = cv2.line(img, (bucket_bin_w[ii], test[3] - 1), (bucket_bin_w[ii], test[3] + 1), (255, 255, 255), 3)
            img = cv2.line(img, (test[2] - 1, bucket_bin_h[ii]), (test[2] + 1, bucket_bin_h[ii]), (255, 255, 255), 3)
    # show logit
    if index == 0:
        pred_logits = pred_logits.reshape(4, 7)
        center_x = int((bbox[2] + bbox[0]) / 2)
        center_y = int((bbox[3] + bbox[1]) / 2)
        blk = np.zeros(img.shape, np.uint8)
        cv2.rectangle(blk, (center_x - 60, center_y - 40), (center_x + 65, center_y + 40), (255, 255, 255), cv2.FILLED)
        img = cv2.addWeighted(img, 1.0, blk, 0.4, 1)
        title_list = ['x1: ', 'y1: ', 'x2: ', 'y2: ']
        for ii in range(4):
            text = title_list[ii]
            if pred_bucket[ii] > 0:
                text += str(pred_logits[ii][pred_bucket[ii]-1])[:4] + ' '
            else:
                text += '*    '
            text += str(pred_logits[ii][pred_bucket[ii]])[:4] + ' '
            if pred_bucket[ii] < 6:
                text += str(pred_logits[ii][pred_bucket[ii] + 1])[:4] + ' '
            else:
                text += '*    '
            cv2.putText(img, text, (center_x - 50, center_y - 20 + ii * 13), font, 0.35, (0, 0, 0), 1)

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
        img = draw_after_bbox(img, each_pred['gt_bbox'], -1, 2, -1)
        img = draw_after_bbox(img, each_pred['coarse'], each_pred['score'], 0, each_pred['category_id'], each_pred['bucket_logit'], each_pred['bucket_pred'])
        img = draw_after_bbox(img, each_pred['proposal'], -1, 1, -1)
        img = draw_after_bbox(img, each_pred['bbox'], -1, 3, -1)
        break
    plt.imsave('coco_img/%012d.jpg'%img_id, img)
    return True

for each_data in match_list:
    draw_img(each_data)
    
