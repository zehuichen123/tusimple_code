import mmcv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = '/mnt/truenas/scratch/czh/data/future/'
train_anno_path = data_path + 'annotations/train_set.json'

train_data = mmcv.load(train_anno_path)
train_anno = train_data['annotations']
cate_data = train_data['categories']

anno_count = {}
for each_anno in train_anno:
    img_id = each_anno['image_id']
    if img_id in anno_count:
        anno_count[img_id] += 1
    else:
        anno_count[img_id] = 1

anno_count = list(anno_count.values())

print(np.mean(anno_count))
print(max(anno_count))
print(min(anno_count))
