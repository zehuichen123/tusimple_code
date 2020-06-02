import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
pred_path = '/mnt/truenas/scratch/czh/kl_baseline/experiments/faster_r50_1x_l1_varhead_binary_finetune_9head/coco_val2017_result.json'
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
        score = each_pred['score']
        area = (bbox[3] - bbox[1] + 1) * (bbox[2] - bbox[0] + 1)
        select_index = np.where(gt_cate == cate_id)[0]
        if select_index.shape[0] != 0:
            select_gt_bbox = gt_bbox[select_index]
            ovr = compute_iou(bbox, select_gt_bbox)
            max_index = np.argmax(ovr); max_iou = np.max(ovr)
            if max_iou > 0.5:
                match_gt_bbox = select_gt_bbox[max_index]
                match_pair.append([each_pred, match_gt_bbox, max_iou])
    return match_pair

def process_img(img_id):
    match_pair = match_pred_gt(img_id)
    diff_data = []
    var_data = []
    iou_data = []
    for each_pair in match_pair:
        pred_data, match_gt_bbox, iou_score = each_pair
        proposal = pred_data['proposal']
        var = np.array(pred_data['var']).reshape(9, 4)
        bbox = pred_data['bbox']
        score = pred_data['score']
        iou_data.append(iou_score)
        width = proposal[2] - proposal[0] + 1
        height = proposal[3] - proposal[1] + 1
        for ii in range(4):
            if ii % 2 == 0:
                diff_data.append((match_gt_bbox[ii] - bbox[ii] + 1) / width)
                var_data.append(var[:, ii].reshape(1, 9))
            if ii % 2 == 1:
                diff_data.append((match_gt_bbox[ii] - bbox[ii] + 1) / height)
                var_data.append(var[:, ii].reshape(1, 9))
    try:
        var_data = np.vstack(var_data)
    except:
        var_data = np.zeros((0, 9))
    return np.array(diff_data).reshape(-1, 1), var_data, np.array(iou_data).reshape(-1, 1)

import concurrent.futures
diff_list = []; var_list = []; iou_list = []
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    for index, res_data in enumerate(executor.map(process_img, img_ids)):
        diff_data, var_data, iou_data = res_data
        diff_list.append(diff_data)
        var_list.append(var_data)
        iou_list.append(iou_data)
        if index % 1000 == 0:
            print("Processing %d/%d"%(index+1, len(img_ids)))
            
scatter_diff = abs(np.vstack(diff_list))
scatter_var = np.vstack(var_list)
scatter_iou = np.vstack(iou_list)

print(scatter_diff.shape)
print(scatter_var.shape)

# fig = plt.figure(figsize=(7, 7))
# plt.hist(scatter_diff, bins=50)
# score_list = []
# np.set_printoptions(suppress=True, precision=3)
# for ii in range(1, 10):
#     score_list.append(np.percentile(scatter_diff, ii * 10) * 10)

# print(score_list)

fig = plt.figure(figsize=(15, 15))
for ii in range(9):
    fig.add_subplot(3, 3, ii+1)
    coef = np.corrcoef(scatter_var[:, ii].reshape(-1,), scatter_diff.reshape(-1,))[0, 1]
    plt.scatter(scatter_var[:, ii].reshape(-1,), scatter_diff.reshape(-1,), s=3, alpha=0.1)
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    plt.grid(True)
    plt.xlabel('var')
    plt.ylabel('diff')
    plt.title('%d: '%((ii+1)*10) + str(coef)[:8])
fig.savefig('img/binary_9head_var_vs_diff.png')

mean_var = np.mean(scatter_var, axis=-1)
fig = plt.figure(figsize=(5, 5))
plt.scatter(mean_var.reshape(-1,), scatter_diff.reshape(-1,), s=3, alpha=0.1)
coef = np.corrcoef(mean_var.reshape(-1,), scatter_diff.reshape(-1,))[0, 1]
plt.grid(True)
plt.xlabel('var')
plt.ylabel('diff')
plt.title('mean' + str(coef)[:8])
fig.savefig('img/binary_9head_mean_var_vs_diff.png')

# title_list = ['x1', 'y1', 'x2', 'y2']
# fig = plt.figure(figsize=(10, 10))
# for ii in range(4):
#     fig.add_subplot("22"+str(ii+1))
#     var_list = []
#     diff_list = []
#     coef = np.corrcoef(scatter_var[:, ii], scatter_diff[:, ii])[0, 1]
#     plt.scatter(scatter_var[:, ii], scatter_diff[:, ii], s=3, alpha=0.1)
#     plt.xlabel('var')
#     plt.ylabel('diff')
#     # plt.plot([0, 2], [0, 2], c='black')
#     # plt.plot([-0.1, 0], [2, 0], c='black')
#     plt.grid(True)
#     plt.title(title_list[ii] + ' ' +str(coef))
#     # plt.xlim(-0.01, 0.25)
#     # plt.ylim(-0.01, 0.25)
# fig.savefig('img/res50_val_kl.png', dpi=200)


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


