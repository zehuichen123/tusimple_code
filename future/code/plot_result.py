import mmcv
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mutils
from pycocotools.coco import COCO
import cv2

pred_path = './results.pkl'
anno_path = '/home/tusimple/Documents/data/future/annotations/val_set_cat.json'
img_path = '/home/tusimple/Documents/data/future/images/val/'
num_class = 34

pred_data = mmcv.load(pred_path)
anno_data = mmcv.load(anno_path)
coco = COCO(anno_path)

color_list = [[0, 153, 51], [0, 153, 255], [255, 255, 0], [204, 102, 255],\
              [255, 0, 0], [204, 0, 255], [102, 255, 255], [255, 102, 102]]
ratio = 1
def decode_mask(mask_encode):
    mask_decode = mutils.decode(mask_encode)
    return mask_decode

def apply_mask(img, mask, color, alpha=.5):
    for c in range(3):
        img[:, :, c] = np.where(mask==1, img[:, :, c] *
                                (1 - alpha) + alpha * color[c],
                                img[:, :, c])
    return img

def draw_mask(img, mask_data, index):
    global color_list
    index %= 5
    mask_mat = decode_mask(mask_data)
    img = apply_mask(img, mask_mat, color_list[index], .5)
    return img

def draw_bbox(img, bbox_score_data, index, show_score=False):
    global color_list
    index %= 5
    test = list(map(lambda x: int(x), bbox_score_data[:4]))
    img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[index], 2)
    if show_score:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ""
        if show_score:
            text += str(bbox_score_data[4])[:4]
        rec_width = len(text) * 6
        img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
        cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    return img

def parse_img(img, anno):
    for cid in range(num_class):
        if len(anno) == 1:
            bbox_data = [[] for ii in range(num_class)]; mask_data = anno[0]
        else:
            bbox_data = anno[0]; mask_data = anno[1]
        for bbox, mask in zip(bbox_data[cid], mask_data[cid]):
            if bbox[4] > 0.05:
                img = draw_bbox(img, bbox, cid, show_score=True)
                img = draw_mask(img, mask, cid)
    return img

for index, (anno, pred) in enumerate(zip(anno_data['images'], pred_data)):
    img_name = anno['file_name'] + '.jpg'
    img = cv2.imread(img_path + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = parse_img(img, pred)
    save_path = 'save_img/%s' % img_name
    plt.imsave(save_path, img)
    if index % 100 == 0:
        print("Saving %d/%d" % (index, len(pred_data)))