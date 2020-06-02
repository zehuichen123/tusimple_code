import os
import time
import glob
import cv2
import random
import uuid
import pickle as pkl
import numpy as np

import mmcv
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

def parse_file(file_name):
    with open(file_name, 'rb') as f:
        data = f.read()
    data_metrics = metrics_pb2.Objects()
    data_metrics.ParseFromString(data)
    return data_metrics

whole_path = '/mnt/truenas/scratch/czh/waymo/code/ensemble/nms_results/pred_result/train_model_on_test_nb3maxgt400minscore0.03_weightnms_0.5_0.03_linear.bin'
exp_path = '/mnt/truenas/scratch/lqf/code/lancher/simpledet/experiments/ensemble/testing/nms_combine/nb10maxgt400minscore0.05_reweight0.4_0.4_weightednms0.7_0.7_0.5_0.5_0.5_0.5_FINAL_2/pred.bin'

whole_data = parse_file(whole_path)
exp_data = parse_file(exp_path)

veh_list = []; ped_list = []; cyc_list = []
print("Parsing Whole Data")
for obj in whole_data.objects:
    if obj.object.type == 1:
        veh_list.append(obj)
print("Parsnig Exp Data")
for obj in exp_data.objects:
    if obj.object.type == 2:
        ped_list.append(obj)
    if obj.object.type == 4:
        cyc_list.append(obj)
obj_list = veh_list + ped_list + cyc_list

print("Start Saving...")

def _create_pd_file_example(obj_list, save_file=None):
	"""Creates a prediction objects file."""
	objects = metrics_pb2.Objects()
	obj_list = sorted(obj_list, key=lambda o: o.score, reverse=True)
	obj_list = obj_list[:1800_0000]
	for obj in obj_list:
		objects.objects.append(obj)
	# Add more objects. Note that a reasonable detector should limit its maximum
	# number of boxes predicted per frame. A reasonable value is around 400. A
	# huge number of boxes can slow down metrics computation.

	# Write objects to a file.
	if save_file is None:
		f = open('./pred_result/%s'%save_name, 'wb')
	else:
		f = open(save_file, 'wb')
	f.write(objects.SerializeToString())
	f.close()

_create_pd_file_example(obj_list, save_file='/mnt/truenas/scratch/czh/waymo/test_result/train_only.bin')