import os
import os.path as osp
import numpy as np
import json
import cv2
from PIL import Image, ImageFile
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt

from visualizer import VisImage
import pdb

def init_img2ann_dict(json_data):
    im2ann_dict = {}
    for ann_item in json_data['annotations']:
        im_id = ann_item['image_id']
        if im_id not in im2ann_dict.keys():
            im2ann_dict[im_id] = [ann_item,]
        else:
            im2ann_dict[im_id].append(ann_item)

    return im2ann_dict

def annToRLE(ann, i_w, i_h):
    h, w = i_h, i_w
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(ann, i_w, i_h):
    rle = annToRLE(ann, i_w, i_h)
    return maskUtils.decode(rle)

class Future3dDataset():
    
    def __init__(self, json_path, img_dir="images"):
        assert json_path is not None
        with open(json_path, 'r') as f:
            self._json_data = json.load(f)
        self.img_dir = img_dir
        self._im2ann_dict = init_img2ann_dict(self._json_data)
        categories_data = self._json_data['categories']
        self._cate_dict = {_['id']: _ for _ in categories_data}

    def load_im_data(self, im_id):
        im_item = self._json_data['images'][im_id]
        im_name = im_item['file_name']
        im_path = osp.join(self.img_dir, "{}.jpg".format(im_name))
        im = np.asarray(Image.open(im_path))
        return im
    
    def load_im_info(self, im_id):
        return self._json_data['images'][im_id]
    
    def load_anns_by_imid(self, im_id):
        im_item = self._json_data['images'][im_id]
        im_id = im_item['id']
        ann_item_list = self._im2ann_dict[im_id]
        return ann_item_list
    
    def load_ann_by_annid(self, ann_id):
        ann_item = self._json_data['annotations'][ann_id]
        return ann_item
    
    def vis_image(self, im_id):
        img = self.load_im_data(im_id)
        im_item = self.load_im_info(im_id)
        im_w = im_item['width']
        im_h = im_item['height']
        
        vis_img = VisImage(img)

        ann_list = self.load_anns_by_imid(im_id)
        instance_list = []
        for ann_item in ann_list:
            cate_id = ann_item['category_id']
            text = self._cate_dict[cate_id]['fine-grained category name']
            mask = annToMask(ann_item, im_w, im_h)
            vis_img.add_mask(mask, text=text)
            
            instance = ann_item.copy()
            instance['name'] = text
            instance_list.append(instance)
            
        print("[INFO] File Name: {}.jpg, instances: ".format(im_item['file_name']))
        for item in instance_list:
            print("category name: {}, model: {}, texture: {}".format(item['name'], item['model_id'], item['texture_id']))
        
        return vis_img, instance_list
        
if __name__ == "__main__":
    data_path = '/mnt/truenas/scratch/czh/data/future/'
    json_path = data_path + 'annotations/train_set.json'
    img_dir = data_path + 'images/train'
    dataset = Future3dDataset(json_path=json_path, img_dir=img_dir)
    vis_img, ann_list = dataset.vis_image(im_id=923)
    vis_img.save('demo.jpg')
        