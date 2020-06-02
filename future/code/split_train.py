import mmcv
import numpy as np
import time
from tqdm import tqdm
np.random.seed(0)

data_path = '/mnt/truenas/scratch/czh/data/future/'
train_anno = data_path + 'annotations/train_set.json'

t1 = time.time()
train_data = mmcv.load(train_anno)
t2 = time.time()

print("Loading Data in %g s" % (t2 - t1))

category = train_data['categories']
img_data = np.array(train_data['images'])
annotations = train_data['annotations']

data_size = 12144
val_size = 3000
train_size = data_size - val_size

img_ids = np.arange(data_size)
np.random.shuffle(img_ids)

train_ids = img_ids[:train_size]
val_ids = img_ids[train_size:]

# for image annotations
train_img = img_data[train_ids].tolist()
val_img = img_data[val_ids].tolist()

# for instance annotations
train_ids_set = set(train_ids.tolist())
val_ids_set = set(val_ids.tolist())
train_anno_list = []
val_anno_list = []
for each_anno in tqdm(annotations):
    if each_anno['image_id'] - 1 in train_ids_set:
        each_anno['iscrowd'] = False
        train_anno_list.append(each_anno)
    elif each_anno['image_id'] - 1 in val_ids_set:
        each_anno['iscrowd'] = False
        val_anno_list.append(each_anno)
    else:
        print("SHOULD NOT REACH HERE!")
        exit()
assert len(train_anno_list) + len(val_anno_list) == len(annotations)

new_train = {
    'images': train_img,
    'categories': category,
    'annotations': train_anno_list
}
new_val = {
    'images': val_img,
    'categories': category,
    'annotations': val_anno_list
}

t3 = time.time()
print("Parsing Data in %g s" % (t3 - t2))

mmcv.dump(new_train, data_path + 'annotations/train_new.json')
mmcv.dump(new_val, data_path + 'annotations/val_new.json')

t4 = time.time()
print("Saving Data in %g s" % (t4 - t3))
