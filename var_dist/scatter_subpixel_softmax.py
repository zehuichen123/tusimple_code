import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
pred_path = '/mnt/truenas/scratch/czh/simpledet_tmp2/experiments/sabl_r50v1_fpn_1x_weight_norm/coco_val2017_result.json'
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
        coarse = each_pred['bbox']
        score = each_pred['score']
        select_index = np.where(gt_cate == cate_id)[0]
        if select_index.shape[0] != 0:
            select_gt_bbox = gt_bbox[select_index]
            ovr = compute_iou(coarse, select_gt_bbox)
            max_index = np.argmax(ovr); max_iou = np.max(ovr)
            if max_iou > 0.5:
                match_gt_bbox = select_gt_bbox[max_index]
                match_pair.append([each_pred, match_gt_bbox, max_iou])
    return match_pair

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def process_bucket_logits(bbox_bucket_logit):
    bbox_bucket_logit = np.array(bbox_bucket_logit).reshape(4, 8)
    # mean = np.min(bbox_bucket_logit, axis=0)
    # bbox_bucket_logit -= mean
    # bbox_bucket_logit = softmax(bbox_bucket_logit)
    emtpy_mat = np.zeros((4, 1))
    sec_mat = np.concatenate([emtpy_mat, bbox_bucket_logit[:, :-1]], axis=1)
    assert sec_mat.shape == (4, 8)
    sum_mat = bbox_bucket_logit + sec_mat

    bucket_ind = (np.argmax(sum_mat, axis=-1) - 1).astype(np.int32)
    assert bucket_ind.shape == (4,)
    near_bucket_ind = bucket_ind + 1
    batch_ind = np.arange(4)
    bucket_logit = bbox_bucket_logit[batch_ind, bucket_ind]
    near_bucket_logit = bbox_bucket_logit[batch_ind, near_bucket_ind]
    sum_logit = bucket_logit + near_bucket_logit

    pos_ind = near_bucket_logit > bucket_logit
    neg_ind = near_bucket_logit <= bucket_logit
    bucket_ind[pos_ind] += 1
    # return bbox_bucket_logit[batch_ind, bucket_ind]
    return sum_logit
    # res = sum_logit / np.sum(bbox_bucket_logit, axis=-1)
    # return res

def process_img(img_id):
    match_pair = match_pred_gt(img_id)
    diff_data = [[] for i in range(4)]
    delta_data = [[] for i in range(4)]
    var_data = [[] for i in range(4)]
    iou_data = []
    argmax_bucket_list = []
    for each_pair in match_pair:
        pred_data, match_gt_bbox, iou_score = each_pair
        coarse = pred_data['coarse']
        bbox = pred_data['bbox']
        score = pred_data['score']
        bucket_logit = pred_data['bucket_logit']

        argmax_bucket = np.argmax(np.array(bucket_logit).reshape(-1, 8), axis=-1)
        argmax_bucket_list.append(argmax_bucket)

        tmp_bucket_logit = process_bucket_logits(bucket_logit)

        bucket_logit = pred_data['coarse']

        argmax_bucket = np.argmax(np.array(bucket_logit).reshape(-1, 8), axis=-1)
        argmax_bucket_list.append(argmax_bucket)

        bucket_logit = process_bucket_logits(bucket_logit)

        iou_data.append(iou_score)
        proposal = pred_data['proposal']
        width = (proposal[2] - proposal[0] + 1) / 14
        height = (proposal[3] - proposal[1] + 1) / 14
        for ii in range(4):
            if ii % 2 == 0:
                diff_data[ii].append((match_gt_bbox[ii] - bbox[ii]) / width)
                var_data[ii].append(bucket_logit[ii])
                delta_data[ii].append(tmp_bucket_logit[ii])

            else:
                diff_data[ii].append((match_gt_bbox[ii] - bbox[ii]) / height)
                var_data[ii].append(bucket_logit[ii])
                delta_data[ii].append(tmp_bucket_logit[ii])

    # return np.array(diff_data).transpose(), np.array(var_data).transpose(), np.array(argmax_bucket_list).reshape(-1, 1)
    return np.array(diff_data).transpose(), np.array(var_data).transpose(), np.array(delta_data).transpose()

import concurrent.futures
diff_list = []; var_list = []; pre_list = []
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    for index, res_data in enumerate(executor.map(process_img, img_ids)):
        diff_data, var_data, pre_data = res_data
        diff_list.append(diff_data)
        var_list.append(var_data)
        pre_list.append(pre_data)
        if index % 1000 == 0:
            print("Processing %d/%d"%(index+1, len(img_ids)))
            
scatter_diff = abs(np.vstack(diff_list))
scatter_var = abs(np.vstack(var_list))
scatter_pre = abs(np.vstack(pre_list))
# scatter_bucket = np.vstack(bucket_list)
# scatter_bucket = scatter_bucket.astype(np.int32)
# pred_data = np.bincount(scatter_bucket.reshape(-1, ))

# np.save('data/diff.npy', scatter_diff)
# np.save('data/var.npy', scatter_var)
# np.save('data/pre.npy', scatter_pre)
# exit()

print(scatter_diff.shape)
print(scatter_var.shape)

# title_list = ['x1', 'y1', 'x2', 'y2']
# fig = plt.figure(figsize=(20, 10))
# cm = plt.get_cmap('bwr')
# bad_list = []
# for ii in range(4):
#     fig.add_subplot(2, 4, ii+1)
#     coef = np.corrcoef(1 - scatter_var[:, ii], scatter_diff[:, ii])[0, 1]
#     plt.scatter(1 - scatter_var[:, ii], scatter_diff[:, ii], s=5, alpha=0.05)
#     plt.xlabel('logit')
#     plt.ylabel('diff')
#     # plt.plot([0, 2], [0, 2], c='black')
#     # plt.plot([-0.1, 0], [2, 0], c='black')
#     plt.grid(True)
#     plt.title(title_list[ii] + ' ' +str(coef))
#     # plt.xlim(-0.01, 0.9)
#     plt.ylim(-0.01, 2.)

title_list = ['x1', 'y1', 'x2', 'y2']
fig = plt.figure(figsize=(20, 10))
for ii in range(4):
    fig.add_subplot(2, 4, ii+1)
    coef = np.corrcoef(scatter_pre[:, ii], scatter_diff[:, ii])[0, 1]
    plt.scatter(scatter_pre[:, ii],scatter_diff[:, ii], \
                        s=5, alpha=0.1)
    # print(min(scatter_var[:, ii]))
    plt.xlabel('score')
    plt.ylabel('diff')
    # plt.plot([0, 2], [0, 2], c='black')
    # plt.plot([-0.1, 0], [2, 0], c='black')
    plt.grid(True)
    plt.title(title_list[ii] + ' ' +str(coef))
    # plt.xlim(-0.01, 1.05)
    plt.ylim(-0.01, 2)

title_list = ['x1', 'y1', 'x2', 'y2']
for ii in range(4):
    fig.add_subplot(2, 4, ii+1+4)
    coef = np.corrcoef(scatter_var[:, ii], scatter_diff[:, ii])[0, 1]
    plt.scatter(scatter_var[:, ii],scatter_diff[:, ii], \
                        s=5, alpha=0.1)
    # print(min(scatter_var[:, ii]))
    plt.xlabel('prescore')
    plt.ylabel('diff')
    # plt.plot([0, 2], [0, 2], c='black')
    # plt.plot([-0.1, 0], [2, 0], c='black')
    plt.grid(True)
    plt.title(title_list[ii] + ' ' +str(coef))
    # plt.xlim(-0.01, 1.05)
    plt.ylim(-0.01, 2)

fig.savefig('img/subpixel_softmax_score_vs_diff.png', dpi=200)

# title_list = ['x1', 'y1', 'x2', 'y2']
# fig = plt.figure(figsize=(20, 10))
# for ii in range(4):
#     fig.add_subplot(2, 4, ii+1+4)
#     # coef = np.corrcoef(scatter_var[:, ii], scatter_diff[:, ii])[0, 1]
#     plt.hist(scatter_var[:, ii] - np.min(scatter_var[:, ii]), bins=50, range=(0, 1), alpha=0.3)
#     # print(min(scatter_var[:, ii]))
#     plt.xlabel('prescore')
#     plt.ylabel('ratio')
#     # plt.plot([0, 2], [0, 2], c='black')
#     # plt.plot([-0.1, 0], [2, 0], c='black')
#     plt.grid(True)
#     # plt.title(title_list[ii] + ' ' +str(coef))
#     # plt.xlim(-0.01, 0.9)
#     # plt.ylim(-0.01, 4)

# fig.savefig('img/subpixel_softmax_prescore_ratio', dpi=200)