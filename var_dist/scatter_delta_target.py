import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
pred_path = '/mnt/truenas/upload/czh/simpledet_baseline4/experiments/faster_r50_1x_paper_l1_varhead_2fc_gs1_woexp/coco_val2017_result.json'
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
                match_pair.append([each_pred, match_gt_bbox, max_iou])
    return match_pair

def process_img(img_id):
    match_pair = match_pred_gt(img_id)
    diff_data = [[] for i in range(4)]
    delta_data = [[] for i in range(4)]
    diff_2nd_data = [[] for i in range(4)]
    var_data = [[] for i in range(4)]
    iou_data = []
    for each_pair in match_pair:
        pred_data, match_gt_bbox, iou_score = each_pair
        proposal = pred_data['proposal']
        delta = pred_data['delta']
        bbox = pred_data['bbox']
        score = pred_data['score']
        var = pred_data['var']
        iou_data.append(iou_score)
        width = proposal[2] - proposal[0] + 1
        height = proposal[3] - proposal[1] + 1
        for ii in range(4):
            if ii % 2 == 0:
                diff_2nd_data[ii].append((match_gt_bbox[ii] - bbox[ii]) / width)
                diff_data[ii].append((match_gt_bbox[ii] - proposal[ii]) / width)
            else:
                diff_2nd_data[ii].append((match_gt_bbox[ii] - bbox[ii]) / height)
                diff_data[ii].append((match_gt_bbox[ii] - proposal[ii]) / height)
            delta_data[ii].append(delta[ii])
            var_data[ii].append(var[ii])
    return np.array(diff_data).transpose(), np.array(delta_data).transpose(),\
            np.array(diff_2nd_data).transpose(), np.array(var_data).transpose()

import concurrent.futures
diff_list = []; delta_list = []; var_list = []; diff_2nd_list = []
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    for index, res_data in enumerate(executor.map(process_img, img_ids)):
        diff_data, delta_data, diff_2nd_data, var_data= res_data
        diff_list.append(diff_data)
        delta_list.append(delta_data)
        var_list.append(var_data)
        diff_2nd_list.append(diff_2nd_data)
        if index % 1000 == 0:
            print("Processing %d/%d"%(index+1, len(img_ids)))
            
scatter_diff = abs(np.vstack(diff_list))
scatter_delta = abs(np.vstack(delta_list))
scatter_var = abs(np.vstack(var_list))
scatter_diff_2nd = abs(np.vstack(diff_2nd_list))

# np.save('data/diff.npy', scatter_diff)
# np.save('data/var.npy', scatter_var)
# np.save('data/iou.npy', scatter_iou)

print(scatter_diff.shape)
print(scatter_delta.shape)

title_list = ['x1', 'y1', 'x2', 'y2']
fig = plt.figure(figsize=(20, 10))
cm = plt.get_cmap('bwr')
bad_list = []
for ii in range(4):
    fig.add_subplot(2, 4, ii+1)
    bad_index = abs(scatter_delta[:, ii] * 0.1 - scatter_diff[:, ii]) > 0.1
    coef = np.corrcoef(scatter_delta[:, ii] * 0.1, scatter_diff[:, ii])[0, 1]
    plt.scatter(scatter_delta[:, ii] * 0.1, scatter_diff[:, ii], s=5,\
                    alpha=0.05, cmap=cm, c=bad_index)
    bad_list.append(bad_index)
    plt.xlabel('delta')
    plt.ylabel('diff')
    # plt.plot([0, 2], [0, 2], c='black')
    # plt.plot([-0.1, 0], [2, 0], c='black')
    plt.grid(True)
    plt.title(title_list[ii] + ' ' +str(coef))
    plt.xlim(-0.01, 0.9)
    plt.ylim(-0.01, 0.9)

for ii in range(4):
    fig.add_subplot(2, 4, ii+1+4)
    coef = np.corrcoef(scatter_var[:, ii] * 0.1, scatter_diff_2nd[:, ii])[0, 1]
    plt.scatter(scatter_var[:, ii] * 0.1,scatter_diff_2nd[:, ii], \
                        s=5, alpha=0.05, cmap=cm, c=bad_list[ii])
    plt.xlabel('var')
    plt.ylabel('diff_2nd')
    # plt.plot([0, 2], [0, 2], c='black')
    # plt.plot([-0.1, 0], [2, 0], c='black')
    plt.grid(True)
    plt.title(title_list[ii] + ' ' +str(coef))
    plt.xlim(-0.01, 0.9)
    plt.ylim(-0.01, 0.9)

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


