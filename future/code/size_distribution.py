import mmcv
import matplotlib.pyplot as plt
import numpy as np

data_path = '/mnt/truenas/scratch/czh/data/future/'
train_anno_path = data_path + 'annotations/train_set.json'

train_data = mmcv.load(train_anno_path)
train_anno = train_data['annotations']
cate_data = train_data['categories']
print("Loading Done")

# size_count = [0 for ii in range(3)]
# for each_anno in train_anno:
#     x, y, w, h = each_anno['bbox']
#     size = w * h
#     if size < 113 ** 2:
#         size_count[0] += 1
#     elif size > 113 ** 2 and size <= 256 ** 2:
#         size_count[1] += 1
#     else:
#         size_count[2] += 1
area_list = []
for each_anno in train_anno:
    x, y, w, h = each_anno['bbox']
    area_list.append((w * h) ** 0.5)
area_data = np.array(area_list)

for ii in range(5):
    thres = (ii + 1) * 0.2 * 100
    val = np.percentile(area_data, thres)
    print("Percentile %d: %g" % (thres, val))

fig = plt.figure(figsize=(5, 5))
plt.hist(area_data, bins=50)
# x_list = [ii for ii in range(3)]
# plt.bar(x_list, size_count)
fig.savefig('../size_distribution.png')

# [4243, 24533, 32234]
