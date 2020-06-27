import mmcv
data_path = '/mnt/truenas/scratch/czh/data/future/'
# pred_path = '/mnt/truenas/scratch/czh/mmdet/work_dirs/submit_solo_x101_dcn_gcb_4x_mst_fp16'
pred_path = '/mnt/truenas/scratch/czh/mmdet2.0/work_dirs/submit_htc_res2_fpn_4x_gcb_dcn_mst_largemaskalign_mscore_fp16_wosyncbn'

pred_data = mmcv.load(pred_path + '/results.pkl')
val_anno = mmcv.load(data_path + '/annotations/val_set.json')

num_class = 34
segm_only = False
add_bbox = False

anno_list = []
for anno, pred in zip(val_anno['images'], pred_data):
    if segm_only:
        mask_data = pred; score_data = [[] for ii in range(num_class)]
        bbox_data = [[] for ii in range(num_class)]
    else:
        mask_info = pred[1]; bbox_data = pred[0]
        if len(mask_info) == 2:
            mask_data = mask_info[0]; score_data = mask_info[1]
        else:
            mask_data = mask_info; score_data = [[] for ii in range(num_class)]
    for cid in range(num_class):
        bboxes = bbox_data[cid]
        masks = mask_data[cid]
        scores = score_data[cid]

        for ii in range(len(masks)):
            if segm_only:
                mask = masks[ii][0]; score = masks[ii][1]
            else:
                x1, y1, x2, y2, score = bboxes[ii]
                # x = (x1 + x2) / 2; y = (y1 + y2) / 2
                w = x2 - x1 + 1; h = y2 - y1 + 1
                mask = masks[ii]
            ms_score = 1.0 if len(scores) == 0 else scores[ii]
            final_score = ms_score * score
            res = {
                'image_id': anno['id'],
                'category_id': cid + 1,
                'segmentation': {'size': mask['size'], 'counts': mask['counts'].decode('utf-8')},
                'score': float('%.3f' % final_score),
            }
            if add_bbox:
                res['bbox'] = [float(ii) for ii in [x1, y1, w, h]]
            anno_list.append(res)

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

mmcv.dump(submit, '../segmentation_results.json')