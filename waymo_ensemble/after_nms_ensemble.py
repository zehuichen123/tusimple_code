import os
import time
import glob
import numpy as np
import pickle as pkl
from utils.nms import py_nms_wrapper, matrix_nms_wrapper, cython_soft_nms_wrapper, wnms_wrapper

ensemble_list = [
    'past/nb10maxgt400minscore0.05_reweight0.4_0.4_weightednms0.7_0.7_0.5_0.5_0.5_0.5',
    'past/val_new_nb3maxgt400minscore0.03_reweight_weightnms_0.5_0.03_linear',
]

# def get_post_nms_file(filename):
#     fold_name = os.listdir('experiments/%s/validation/'%filename)[0]
#     file_list = glob.glob('experiments/%s/validation/%s/*.pkl' %(filename, fold_name))
#     return file_list
def get_nms_file(path):
    file_list = glob.glob(path + '/*.pkl')
    return file_list

t1 = time.time()

num_ensemble = len(ensemble_list)
res_list = [[] for ii in range(num_ensemble)]
file_list = [get_nms_file(filename) for filename in ensemble_list]
for ii in range(num_ensemble):
    files = file_list[ii]
    for file in files:
        pkl_data = pkl.load(open(file, 'rb'), encoding='latin1')
        res_list[ii] += list(pkl_data.items())

t2 = time.time()
print("Grouping NMS results in %g s" % (t2 - t1))

# NOTE for distributed testing bug
res_list = [dict(res) for res in res_list]
img_list = []
for ii in range(num_ensemble):
    img_list += list(res_list[ii].keys())
img_list_all = list(set(img_list))

# nms_veh = cython_soft_nms_wrapper(0.7)
# nms_ped = cython_soft_nms_wrapper(0.5)
nms_veh = py_nms_wrapper(0.7)
nms_ped = py_nms_wrapper(0.5)
nms_cyc = py_nms_wrapper(0.5)
nms_method_list = [nms_veh, nms_ped, nms_cyc]
nms_mapping = {
    'TYPE_VEHICLE': 0,
    'TYPE_PEDESTRIAN': 1,
    'TYPE_CYCLIST': 2,
}

rescore_list = [[0.9, 1.0], [1.0, 0.9], [1.0, 0.5]]

def parse_img(param):
    start, end = param
    img_list = img_list_all[start:end]
    rec_list = []
    for img_url in img_list:
        ensemble_list = [res_list[ii][img_url] for ii in range(num_ensemble) if img_url in res_list[ii]]

        cate_name = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
        rec_dict = {}
        rec_dict['meta_info'] = ensemble_list[0]['meta_info']   # directly get meta_info from 1st candidate

        det_dict = {}
        for each_cate in cate_name:
            nms_list = []
            for ii in range(len(ensemble_list)):
                nms_res = ensemble_list[ii]['det_xyxys'][each_cate]
                if type(nms_res) == list or nms_res.shape[0] == 0:
                    nms_res = np.zeros((0, 5))
                nms_res[:, 4] *= rescore_list[nms_mapping[each_cate]][ii]
                nms_list.append(nms_res)
            dets = np.vstack(nms_list)
            nms = nms_method_list[nms_mapping[each_cate]]  # NOTE select nms for different category
            dets = nms(dets)
            det_dict[each_cate] = dets
        rec_dict['det_xyxys'] = det_dict
        rec_list.append(rec_dict)
    return rec_list
# =====================================================================
print("Start Parsing...")       # nms img
t1 = time.time()
ensemble_res = []

def generate_batch(num_threads, num_tasks):
    param_list = []
    avg_tasks = (num_tasks + num_threads - 1) // num_threads
    for ii in range(num_threads):
        param_list.append([avg_tasks * ii, avg_tasks * (ii + 1)])
    return param_list

param_list = generate_batch(50, len(img_list_all))
import concurrent.futures
with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
    for index, res in enumerate(executor.map(parse_img, param_list)):
        ensemble_res += res
t2 = time.time()
print("Parsing Done in %g s" % (t2 - t1))

# =====================================================================
from nms_results.create_prediction_file_validation import _create_bbox_prediction, _create_pd_file_example
def parse_img_2_obj(param):
    start, end = param
    img_rec = ensemble_res[start:end]
    object_all = []
    for pred_data in img_rec:
        object_list = []
        meta_info = pred_data['meta_info']
        det_xyxys = pred_data['det_xyxys']
        marco_ts = meta_info['timestamp_micros']
        time_of_day = meta_info['time_of_day']
        name = meta_info['name']
        camera_name = meta_info['camera_name']
        for cate in det_xyxys:
            dets = det_xyxys[cate]
            for each_det in dets:
                o = _create_bbox_prediction(each_det, cate, name, marco_ts, camera_name, time_of_day)
                object_list.append(o)
        object_list = sorted(object_list, key=lambda x: x.score, reverse=True)[:400]
        object_all += object_list
    return object_all

object_list = []
t1 = time.time()
print("Convert to .bin...")
param_list = generate_batch(50, len(ensemble_res))

with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
    for index, obj_data in enumerate(executor.map(parse_img_2_obj, param_list)):
        object_list += obj_data
t2 = time.time()
print("Parsing Prediction Data Using: %g s" % (t2 - t1))
_create_pd_file_example(object_list, './nms_results/pred_result/whole_exp_ensemble.bin')
t3 = time.time()
print("Saving Object List Using %g s" % (t3 - t2))



