import mmcv
from pycocotools.coco import COCO

data_path = '/mnt/truenas/scratch/czh/data/future/'
train_anno_path = data_path + 'annotations/train_new.json'
val_anno_path = data_path + 'annotations/val_new.json'

train_data = mmcv.load(train_anno_path)
val_data = mmcv.load(val_anno_path)

for ii in range(len(train_data['annotations'])):
    train_data['annotations'][ii]['iscrowd'] = False

for ii in range(len(val_data['annotations'])):
    val_data['annotations'][ii]['iscrowd'] = False

mmcv.dump(train_data, train_anno_path)
mmcv.dump(val_data, val_anno_path)