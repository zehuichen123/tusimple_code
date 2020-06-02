import mmcv
import numpy as np
import pycocotools.mask as mutils
from nms import py_nms_wrapper

data_path = '/mnt/truenas/scratch/czh/data/future/'
pred_path = '/mnt/truenas/scratch/czh/mmdet/work_dirs/solo_x101_dcn_4x_mst'

pred_data = mmcv.load(pred_path + '/results.pkl')
val_anno = mmcv.load(data_path + '/annotations/val_set.json')

num_class = 34
num_super_class = 7
segm_only = True
anno_list = []

def get_super_category(cate_data):
    super_dict = {}
    for each_cat in cate_data:
        super_name = each_cat['category_name']
        if super_name not in super_dict:
            super_dict[super_name] = [each_cat['id'] - 1]
        else:
            super_dict[super_name].append(each_cat['id'] - 1)
    return super_dict

super_dict = get_super_category(val_anno['categories'])
nms = py_nms_wrapper(0.5)


def parse_img(index_num):
    anno_list = []
    anno, pred = val_anno['images'][index_num], pred_data[index_num]
    if segm_only:
        mask_data = np.array(pred); bbox_data = np.array([[] for ii in range(num_class)])
    else:
        mask_data = pred[1]; bbox_data = pred[0]
    for index, super_name in enumerate(super_dict):
        group_cids = super_dict[super_name]
        bboxes_list = bbox_data[group_cids]; masks_list = mask_data[group_cids]
        all_bboxes = []; all_masks = []; all_cids = []; all_scores = []
        for idx, cid in enumerate(group_cids):
            if segm_only:
                masks = masks_list[idx]
                all_cids.append(np.ones((len(masks), 1)))
                _scores = []; _masks = []; _bboxes = []
                for mask in masks:
                    _mask = mask[0]; _score = mask[1]
                    _scores.append(_score)
                    _masks.append(_mask)
                    decode_mask = mutils.decode(_mask)
                    # line
                    line_sum = np.sum(decode_mask, axis=1)
                    col_sum = np.sum(decode_mask, axis=0)
                    assert line_sum.shape[0] == decode_mask.shape[1]
                    line_idx = np.where(line_sum > 0)[0]
                    col_idx = np.where(col_sum > 0)[0]
                    x1, x2 = line_idx[0], line_idx[-1]
                    y1, y2 = col_idx[0], col_idx[-1]
                    _bboxes.append([x1, y1, x2, y2])
                if len(_bboxes) == 0:
                    continue
                all_scores.append(np.array(_scores).reshape(-1, 1))
                all_cids.append(np.ones((len(masks), 1)) * cid)
                all_masks.append(np.array(_masks).reshape(-1, 1))
                all_bboxes.append(np.array(_bboxes))
            else:
                pass
        # all_bboxes, all_cids, all_masks, all_scores
        if len(all_bboxes) == 0:
            continue
        all_bboxes = np.vstack(all_bboxes)
        all_cids = np.vstack(all_cids)
        all_masks = np.vstack(all_masks)
        all_scores = np.vstack(all_scores)

        all_det = np.concatenate([all_bboxes, all_scores], axis=1)
        keep_idx = nms(all_det)
        all_bboxes = all_bboxes[keep_idx]
        all_cids = all_cids[keep_idx]
        all_masks = all_masks[keep_idx]
        all_scores = all_scores[keep_idx]
        
        for ii in range(all_cids.shape[0]):
            mask = all_masks[ii][0]
            res = {
                'image_id': anno['id'],
                'category_id': int(all_cids[ii]) + 1,
                'segmentation': {'size': mask['size'], 'counts': mask['counts'].decode('utf-8')},
                'score': float(all_scores[ii]),
            }
            anno_list.append(res)
    return anno_list

param_list = range(len(pred_data))
anno_list = []
import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=70) as executor:
        for index, res_data in enumerate(executor.map(process_img, param_list)):
            anno_list += res_data

img_list = []
for img_info in val_anno['images']:
    res = {
        'image_id': img_info['id'],
        'width': img_info['width'],
        'height': img_info['height'],
        'file_name': img_info['file_name']
    }
    img_list.append(res)

submit = {}
submit['images'] = img_list
submit['annotations'] = anno_list

mmcv.dump(submit, 'segmentation_results.json')
