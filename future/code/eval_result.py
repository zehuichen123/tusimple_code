import mmcv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
data_path = '/mnt/truenas/scratch/czh/data/future'
pred_path = '/mnt/truenas/scratch/czh/future/custom_val.json'
anno_path = data_path + '/annotations/val_new.json'

coco_result = mmcv.load(pred_path)['annotations']
coco = COCO(anno_path)

coco_dt = coco.loadRes(coco_result)
coco_eval = COCOeval(coco, coco_dt)
coco_eval.params.iouType = "bbox"
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# coco_eval = COCOeval(coco, coco_dt)
# coco_eval.params.iouType = "segm"
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()