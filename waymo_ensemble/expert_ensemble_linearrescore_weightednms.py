import os
import time
import glob
import math
import cv2
import random
import pprint
from core.detection_module import DetModule
from core.detection_input import Loader
from utils.load_model import load_checkpoint
from utils.patch_config import patch_config_as_nothrow
from functools import reduce
from queue import Queue
from threading import Thread
import argparse
import importlib
import mxnet as mx
import numpy as np
import pickle as pkl
from collections import OrderedDict
import gc
from operator_py.nms import py_nms_wrapper
from operator_py.nms import cython_soft_nms_wrapper, py_weighted_nms
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    parser.add_argument('--save-name', default='nms_combine', help='save name', type=str)  # NOTE
    parser.add_argument('--thresh', help='test score thresh', type=float, default=0.03)
    parser.add_argument('--constrain', default=True, action='store_true')
    parser.add_argument('--NB', help='max num of det bbox per img', type=int, default=400)
    # parser.add_argument('--filter_', default=False, action='store_true')
    parser.add_argument('--vehicle_ensemble', default=True, action='store_true')
    parser.add_argument('--pTest_nms_thr', help='', type=float, default=0.5)  # NOTE
    parser.add_argument('--nb_ensemble', help='', type=int, default=-1)  # NOTE
    parser.add_argument('--reweight_score', default=True, action='store_true')
    parser.add_argument('--noafter_score_filter', default=False, action='store_true')
    parser.add_argument('--softnms', default=False, action='store_true')
    parser.add_argument('--weightednms', default=False, action='store_true')
    parser.add_argument('--sigma', default=0.5, type=float)
    parser.add_argument('--score_thresh', default=0.001, type=float)
    parser.add_argument('--method', default='linear', type=str)
    parser.add_argument('--reweight_minthr', default=0.5, type=float)
    parser.add_argument('--thresh_vehicle_lo',default=0.7, type=float)
    parser.add_argument('--thresh_vehicle_hi',default=0.7, type=float)
    parser.add_argument('--thresh_ped_lo',default=0.5, type=float)
    parser.add_argument('--thresh_ped_hi',default=0.5, type=float)
    parser.add_argument('--thresh_cyc_lo',default=0.5, type=float)
    parser.add_argument('--thresh_cyc_hi',default=0.5, type=float)
    parser.add_argument('--test_mode',default=False, action='store_true')

    args = parser.parse_args()
    return args


def filter(final_dets, NB=400):
    all_dets = []
    for k, det in final_dets.items():
        for det_s in det:
            all_dets.append(det_s.tolist() + [k])
    assert len(all_dets[0]) == 6
    all_dets = sorted(all_dets, key=lambda x: x[-2], reverse=True)
    # print(all_dets)
    assert len(all_dets)>NB
    all_dets = all_dets[:NB]
    final_dets_new = {}
    for det in all_dets:
        key = det[-1]
        if key not in final_dets_new:
            final_dets_new[key] = []
        final_dets_new[key].append(det[:-1])
    for k, v in final_dets_new.items():
        final_dets_new[k] = np.array(v)
        assert final_dets_new[k].shape[0] > 0 and final_dets_new[k].shape[1] == 5
    return final_dets_new


if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = os.environ.get("MXNET_CUDNN_AUTOTUNE_DEFAULT", "0")

    args = parse_args()
    image_set_name = 'validation'
    if args.test_mode:
        image_set_name = 'testing'
    
    # setting
    parent_cat_ids = {
    0: 'TYPE_VEHICLE', 
    1: 'TYPE_PEDESTRIAN', 
    2: 'TYPE_CYCLIST'}
    min_det_score = args.thresh # or pTest.min_det_score  # NOTE
    print('min_det_score: {}'.format(min_det_score))


    # candidate_files_val_l2map={
    #     'cascade_hrfpnw18_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_1600_calibrate3stages':(64.41,76.28,55.97),  # added
    #     'cascade_hrfpnw32_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_1600_calibrate3stages':(65.93,77.58,58.24),
    #     'cascade_hrfpnw48_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_1600_calibrate3stages':(66.40,77.86,56.38),
    #     'cascade_resnext50_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_1600_calibrate3stages':(65.45,76.75,54.66),
    # }
    candidate_files_val_l2map = {
        'cascade_hrfpnw18_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_1280_2240_calibrate3stages_merge':(65.16,76.98,57.32),
        'cascade_hrfpnw32_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_1280_2240_calibrate3stages_merge':(66.49,78.23,59.94),
        'cascade_hrfpnw48_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_1280_2240_calibrate3stages_merge':(66.94,78.45,58.00),
        'cascade_resnext50_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_1600_calibrate3stages_merge':(66.42,77.55, 57.10),
    }

    if args.test_mode:
        # candidate_files_val_l2map={
        # 'cascade_hrfpnw18_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_flip_test_calibrate3stages':(64.41,76.28,55.97), 
        # 'cascade_hrfpnw32_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_flip_test_calibrate3stages':(65.93,77.58,58.24),
        # 'cascade_hrfpnw48_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_flip_test_calibrate3stages':(66.40,77.86,56.38),
        # 'cascade_resnext50_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_test': (65.45,76.75,54.66),
        # }
        candidate_files_val_l2map={
            'cascade_hrfpnw18_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_flip_test_calibrate3stages_merge': (65.16,76.98,57.32),
            'cascade_hrfpnw32_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_flip_test_calibrate3stages_merge':(66.49,78.23,59.94),
            'cascade_hrfpnw48_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_flip_test_calibrate3stages_merge':(66.94,78.45,58.00),
            'cascade_resnext50_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_test_merge':(66.42,77.55, 57.10),
        }

    nb_ensemble=len(candidate_files_val_l2map) if args.nb_ensemble<=0 else args.nb_ensemble
    print('nb_ensemble:{} reweight_score:{}'.format(nb_ensemble, args.reweight_score))
    assert len(candidate_files_val_l2map)>=nb_ensemble
    for l2map in candidate_files_val_l2map.values():
        assert len(l2map) == 3
    val_vehicle_l2map_topk = OrderedDict(sorted(candidate_files_val_l2map.items(), key=lambda x: x[1][0], reverse=True)[:nb_ensemble])
    val_ped_l2map_topk = OrderedDict(sorted(candidate_files_val_l2map.items(), key=lambda x: x[1][1], reverse=True)[:nb_ensemble])
    val_cyc_l2map_topk = OrderedDict(sorted(candidate_files_val_l2map.items(), key=lambda x: x[1][2], reverse=True)[:nb_ensemble])
    if args.test_mode:
        vehicle_candidate_files=[glob.glob('experiments_cyc_expert_trainval/test/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_vehicle_l2map_topk.keys()]
        ped_candidate_files=[glob.glob('experiments_cyc_expert_trainval/test/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_ped_l2map_topk.keys()]
        cyc_candidate_files=[glob.glob('experiments_cyc_expert_trainval/test/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_cyc_l2map_topk.keys()]
    else:
        vehicle_candidate_files=[glob.glob('experiments_cyc_expert/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_vehicle_l2map_topk.keys()]
        ped_candidate_files=[glob.glob('experiments_cyc_expert/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_ped_l2map_topk.keys()]
        cyc_candidate_files=[glob.glob('experiments_cyc_expert/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_cyc_l2map_topk.keys()]
    val_score_vehicle = [score[0] for score in val_vehicle_l2map_topk.values()]
    val_score_ped = [score[1] for score in val_ped_l2map_topk.values()]
    val_score_cyc = [score[2] for score in val_cyc_l2map_topk.values()]

    # NOTE reweight score
    score_list = [val_score_vehicle, val_score_ped, val_score_cyc]
    min_reweight_thr = args.reweight_minthr
    # def reweight_score_func()
    score_list_before = score_list
    score_list_reweighted = []
    for reweight_score_percls in score_list:
        max_score = np.max(reweight_score_percls)
        assert max_score == reweight_score_percls[0]
        min_score = np.min(reweight_score_percls)
        assert min_score == reweight_score_percls[-1]
        delta_score = (1.0-min_reweight_thr) / (len(reweight_score_percls)-1)
        reweight_score_percls = [1.0 - i*delta_score for i in range(nb_ensemble)]
        assert np.max(reweight_score_percls) == 1. and reweight_score_percls[0]==1.0
        assert np.min(reweight_score_percls) == reweight_score_percls[-1]
        if np.min(reweight_score_percls) != min_reweight_thr:
            print('WARMING {} vs {}'.format(np.min(reweight_score_percls), min_reweight_thr))
        # assert np.min(reweight_score_percls) == min_reweight_thr and reweight_score_percls[-1] == min_reweight_thr, 
        score_list_reweighted.append(reweight_score_percls)


    score_list = score_list_reweighted
    print('------------------------------------------------------ ')
    print(score_list_before)
    print(score_list)
    print('score_list before and after: ------------------------- ')
    if not args.reweight_score:
        score_list = [[1]*nb_ensemble, [1]*nb_ensemble, [1]*nb_ensemble]  # NOTE NOTE NOTE NOTE
    print('score_list for ensemble: ', score_list)
    del val_vehicle_l2map_topk, val_ped_l2map_topk, val_cyc_l2map_topk

    assert len(vehicle_candidate_files)==len(ped_candidate_files)==len(cyc_candidate_files)
    for i, (vehicle_file, ped_file, cyc_file) in enumerate(zip(vehicle_candidate_files, ped_candidate_files, cyc_candidate_files)):
        # if len(vehicle_file) == 0:
            # print(vehicle_file)
            # print(i, vehicle_candidate_files[i])
        #     pass
        assert len(vehicle_file) == len(ped_file)==len(cyc_file), '{} vs {} vs {}'.format(len(vehicle_file), len(ped_file), len(cyc_file))
    for file_ in [vehicle_candidate_files, ped_candidate_files, cyc_candidate_files]:
        for elem in zip(*file_):
            seg_id = elem[0].split('/')[-1]
            for e in elem:
                assert e.split('/')[-1]==seg_id
    vehicle_candidate_files = [elem for elem in zip(*vehicle_candidate_files)]
    ped_candidate_files = [elem for elem in zip(*ped_candidate_files)]
    cyc_candidate_files = [elem for elem in zip(*cyc_candidate_files)]
    assert len(vehicle_candidate_files) == len(ped_candidate_files) == len(cyc_candidate_files)
    if args.test_mode:
        assert len(vehicle_candidate_files)==149
    else:
        assert len(vehicle_candidate_files)==200

    # parent_file_name = 'faster_r50v1b_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_topkanchor_mst_smallscale_zhk'
    poster='nb{}maxgt{}minscore{}'.format(nb_ensemble, args.NB, min_det_score)
    if args.reweight_score:
        poster += '_reweight{}'.format(args.reweight_minthr)
       
    if args.noafter_score_filter:
        poster += '_noafterscorefilter'
    if args.softnms:
        poster += '_softnms{}_{}_{}'.format(args.sigma, args.score_thresh, args.method)
    elif args.weightednms:
        poster += '_weightednms{}_{}_{}_{}_{}_{}'.format(args.thresh_vehicle_lo,args.thresh_vehicle_hi,args.thresh_ped_lo,args.thresh_ped_hi,args.thresh_cyc_lo,args.thresh_cyc_hi)
    if args.test_mode:
        poster+= '_FINAL'

    # save_path = os.path.join('experiments/ensemble/{}/{}_nbsemble{}'.format(image_set_name, args.save_name, nb_ensemble))
    # if args.test_mode:
    save_path = os.path.join('experiments/ensemble_trainval/merge/linear_reweight/{}/{}/{}/'.format(image_set_name, args.save_name, poster))
    # else:
    #     save_path = os.path.join('experiments/ensemble_trainval/val_new/{}/{}/{}/'.format(image_set_name, args.save_name, poster))
    os.makedirs(save_path, exist_ok=True)
    nms_list = []
    if args.softnms:
        nms = cython_soft_nms_wrapper(args.pTest_nms_thr, sigma=args.sigma, score_thresh=args.score_thresh, method=args.method)
        assert not args.weightednms
    elif args.weightednms:
        nms1 = partial(py_weighted_nms, thresh_lo=args.thresh_vehicle_lo, thresh_hi=args.thresh_vehicle_hi)
        assert not args.softnms
        nms_list.append(nms1)
        nms2 = partial(py_weighted_nms, thresh_lo=args.thresh_ped_lo, thresh_hi=args.thresh_ped_hi)
        nms_list.append(nms2)
        nms3 = partial(py_weighted_nms, thresh_lo=args.thresh_cyc_lo, thresh_hi=args.thresh_cyc_hi)
        nms_list.append(nms3)
    else:
        nms_list = [py_nms_wrapper(0.7)]
        nms_list += [py_nms_wrapper(args.pTest_nms_thr) for _ in range(2)]
    # output_dict_list_allcls=[]
    for i, (elems_vehicle,elems_ped, elems_cyc) in enumerate(zip(vehicle_candidate_files, ped_candidate_files, cyc_candidate_files)):
        assert len(elems_vehicle) ==len(elems_ped)==len(elems_cyc)== nb_ensemble
        for elem_vehicle, elem_ped, elem_cyc in zip(elems_vehicle, elems_ped, elems_cyc):
            assert elem_vehicle.split('/')[-1]==elem_cyc.split('/')[-1]==elem_cyc.split('/')[-1]
        # keys = -1
        output_dict_list = [[pkl.load(open(pkl_seg,'rb')) for pkl_seg in elems_vehicle]]
        output_dict_list.append([pkl.load(open(pkl_seg, 'rb')) for pkl_seg in elems_ped])
        output_dict_list.append([pkl.load(open(pkl_seg, 'rb')) for pkl_seg in elems_cyc])


        t1_s = time.time()
        # val_score_i = score_list[i][j]
        
        # cid_start = 0  # NOTE for ped, cyc
        # if args.vehicle_ensemble:
        #     cid_start = -1  # for vehicle, ped and cyc
            

        def do_nms(k):
            # for j in range(nb_ensemble):
            #     bbox_xyxy = output_dict_list[j][k]["bbox_xyxy"]
            #     cls_score = output_dict_list[j][k]["cls_score"]

            final_dets = {}
            output_dict_ensembeld = {}
            #     assert cls_score.shape[1] == 3
            #     child_score_w=[0.6,0.5,1.1]
            for cid in range(3):
            # for cid in range(parent_cls_score.shape[1]):  # vehicle, ped, cyc 
                # score = parent_cls_score[:, cid]
                bbox_xyxys = [output_dict_list[cid][pklsegid][k]["bbox_xyxy"] for pklsegid in range(nb_ensemble)]
                cls_scores = [output_dict_list[cid][pklsegid][k]["cls_score"][:,cid] for pklsegid in range(nb_ensemble)]
                assert len(bbox_xyxys) == len(cls_scores) == nb_ensemble  # avoid none in bbox_xyxys or cls_scores
                if bbox_xyxys[0].shape[1] != 4:
                    assert bbox_xyxys[0].shape[1] == 16
                    bbox_xyxys = [bbox_xyxy[:,cid*4:(cid+1)*4] for bbox_xyxy in bbox_xyxys]
                reweight_score = [score_list[cid][cand_id] for cand_id in range(nb_ensemble)]
                
                valid_idx_list = [np.where(cls_score_tmp>=min_det_score)[0] for cls_score_tmp in cls_scores]  # hh
                cls_scores = [cls_score_i*reweight_score_i for cls_score_i,reweight_score_i in zip(cls_scores, reweight_score)]
                cls_scores = [cls_score_tmp[valid_idx] for cls_score_tmp, valid_idx in zip(cls_scores, valid_idx_list)]  # hh
                bbox_xyxys = [bbox[valid_idx] for bbox, valid_idx in zip(bbox_xyxys, valid_idx_list)]  # hh
                
                box = np.concatenate(bbox_xyxys, axis=0)
                cls_scores = np.concatenate(cls_scores, axis=0)
                assert np.all(cls_scores>=0) and np.all(cls_scores<=1)  # TODO check

                if not args.noafter_score_filter:
                    valid_inds = np.where(cls_scores >= min_det_score)[0]  # NOTE
                    cls_scores = cls_scores[valid_inds]
                    box = box[valid_inds]
                    

                
                # if args.filter_:
                #     # NOTE NOTE added
                #     h = box[:,3]-box[:,1]
                #     valid_ = np.where(h>9)[0]  # 7
                #     box, cls_scores = box[valid_], cls_scores[valid_]
                #     w = box[:,2]-box[:,0]
                #     valid_ = np.where(w>7)[0]  # 5
                #     box, cls_scores = box[valid_], cls_scores[valid_]
                det = np.concatenate((box, cls_scores.reshape(-1, 1)), axis=1).astype(np.float32)
                # det = nms(det)
                det = nms_list[cid](det)  # NOTE NOTE
                dataest_cid = parent_cat_ids[cid]
                final_dets[dataest_cid] = det
                
            output_dict_ensembeld["det_xyxys"] = final_dets 
            del bbox_xyxys, cls_scores  # TODO check
            cnt = 0
            for _,v in final_dets.items():
                if len(v)>0:
                    cnt += len(v)
            if cnt >args.NB:
                # print('cnt: ', cnt)
                if args.constrain:
                    final_dets = filter(final_dets, NB=args.NB)
                    for cid_ in range(3):  # NOTE NOTE NOTE
                        dataest_cid = parent_cat_ids[cid_]
                        if dataest_cid not in final_dets:
                            final_dets[dataest_cid] = []
                    output_dict_ensembeld["det_xyxys"] = final_dets

                    # cnt = 0
                    # for _,v in final_dets.items():
                    #     if len(v)>0:
                    #         cnt += len(v)
                    # print('cnt after: ', cnt)

            output_dict_ensembeld['meta_info']=output_dict_list[cid][0][k]['meta_info']  # TODO check
            return (k, output_dict_ensembeld)

        from multiprocessing import cpu_count
        from multiprocessing.pool import Pool
        pool = Pool(cpu_count() // 2)
        output_dict_ensembel = pool.map(do_nms, output_dict_list[0][0].keys())  # TODO check
        output_dict_ensembel = dict(output_dict_ensembel)
        pool.close()
        pool.join()
        

        t2_s = time.time()
        print("nms uses: %.1f" % (t2_s - t1_s))
        
        dump_name = elems_vehicle[0].split('/')[-1].replace('_before_nms', '')
        dump_name = os.path.join(save_path, dump_name)
        del output_dict_list
        gc.collect()
        with open(dump_name, "wb") as fout:
            pkl.dump(output_dict_ensembel, fout)
        print("dump result to {}".format(dump_name))