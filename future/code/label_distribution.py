import mmcv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data_path = '/mnt/truenas/scratch/czh/data/future/'
# train_anno_path = data_path + 'annotations/train_set.json'

# train_data = mmcv.load(train_anno_path)
# train_anno = train_data['annotations']
# cate_data = train_data['categories']

# cid2name = {}
# for each_cat in cate_data:
#     cid2name[each_cat['id']] = each_cat['fine-grained category name']

# print("Annotation Num: ", len(train_anno))
# print("Category: ", cid2name)

# class_count = {}
# for each_anno in train_anno:
#     cid = each_anno['category_id']
#     catename = cid2name[cid]
#     if catename not in class_count:
#         class_count[catename] = 1
#     else:
#         class_count[catename] += 1
# print(class_count)

# mmcv.dump(class_count, 'tmp_data/label_count.json')
class_count = mmcv.load('tmp_data/label_count.json')

class_name, count_num = list(class_count.keys()), list(class_count.values())

count_data = pd.DataFrame(
    {
        'name': class_name,
        'count': count_num,
    }
)
count_data = count_data.set_index('name').rename_axis(None)
count_data = count_data.sort_values('count', ascending=False)
count_data.plot.barh(figsize=(15, 25))
plt.grid(True)
plt.savefig('tmp_data/label_count.png', dpi=100)




