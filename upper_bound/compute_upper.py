import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json
from pycocotools.cocoeval import COCOeval
from cython.bbox import bbox_overlaps_cython
from nms import py_nms_wrapper
nms = py_nms_wrapper(0.5)

anno_path = "/mnt/truenas/upload/czh/data/coco/coco/annotations/instances_val2017.json"
# pred_path = "/mnt/truenas/upload/czh/simpledet_msrcnn/experiments/ms_r50v1_fpn_1x/coco_val2017_result.json"
save_path = '/mnt/truenas/scratch/czh/simpledet_xywh/save_data/'

coco = COCO(anno_path)
# with open(pred_path, 'r') as f:
#     coco_pred = json.load(f)

def get_pred_data(img_id):
    cls_score = np.load(save_path + 'data/%d_score.npy' % img_id)
    bbox = np.load(save_path + 'data/%d_bbox.npy' % img_id)
    return cls_score, bbox

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

def match_pred_gt(img_id):
    gt_bbox, gt_cate = get_gt_data(img_id)
    cls_score, bbox_xyxy = get_pred_data(img_id)
    tmp_dict = {}
    final_dets = {}
    for cid in range(cls_score.shape[1]):
        score = cls_score[:, cid]
        if bbox_xyxy.shape[1] != 4:
            cls_box = bbox_xyxy[:, cid * 4:(cid + 1) * 4]
        else:
            cls_box = bbox_xyxy
        valid_inds = np.where(score > 0.05)[0]
        bbox = cls_box[valid_inds]
        score = score[valid_inds]

        # det = np.concatenate((bbox, score.reshape(-1, 1)), axis=1).astype(np.float32)
        # det_index = nms(det)
        # det = det[det_index]
        # dataset_cid = coco.getCatIds()[cid]
        # final_dets[dataset_cid] = det
        dataset_cid = coco.getCatIds()[cid]
        select_index = np.where(gt_cate == dataset_cid)[0]
        if select_index.shape[0] != 0:
            select_gt_bbox = gt_bbox[select_index]
            ovr = bbox_overlaps_cython(bbox.astype(np.float32), select_gt_bbox.astype(np.float32))
            gt_score = np.max(ovr, axis=1)
            det = np.concatenate((bbox, gt_score.reshape(-1, 1)), axis=1).astype(np.float32)
            det_index = nms(det)
            det = det[det_index]
            dataset_cid = coco.getCatIds()[cid]
            final_dets[dataset_cid] = det
    tmp_dict["det_xyxys"] = final_dets
    return (img_id, tmp_dict)

img_ids = coco.getImgIds()
import concurrent.futures
res_list = []
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    for index, res_data in enumerate(executor.map(match_pred_gt, img_ids)):
        res_list.append(res_data)
        if index % 1000 == 0:
            print("Processing %d/%d"%(index+1, len(img_ids)))

output_dict = dict(res_list)

# print(output_dict[5001])
# exit()
coco_result = []
for iid in output_dict:
    result = []
    # print(output_dict[iid])
    for cid in output_dict[iid]["det_xyxys"]:
        det = output_dict[iid]["det_xyxys"][cid]
        if det.shape[0] == 0:
            continue
        scores = det[:, -1]
        xs = det[:, 0]
        ys = det[:, 1]
        ws = det[:, 2] - xs + 1
        hs = det[:, 3] - ys + 1
        result += [
            {'image_id': int(iid),
                'category_id': int(cid),
                'bbox': [float(xs[k]), float(ys[k]), float(ws[k]), float(hs[k])],
                'score': float(scores[k])}
            for k in range(det.shape[0])
        ]
    result = sorted(result, key=lambda x: x['score'])[-100:]
    coco_result += result

# import json
# json.dump(coco_result,
#     open("experiments/{}/{}_result.json".format(pGen.name, pDataset.image_set[0]), "w"),
#     sort_keys=True, indent=2)

coco_dt = coco.loadRes(coco_result)
coco_eval = COCOeval(coco, coco_dt)
coco_eval.params.iouType = "bbox"
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
