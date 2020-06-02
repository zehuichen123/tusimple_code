import os
import time
import glob
import math
import cv2
import random
import pprint
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
from utils.nms import py_nms_wrapper, matrix_nms_wrapper, cython_soft_nms_wrapper, wnms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    # parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--save-name', default='nms_combine', help='save name', type=str)  # NOTE
    parser.add_argument('--thresh', help='test score thresh', type=float, default=0.03)
    parser.add_argument('--constrain', default=True, action='store_true')
    parser.add_argument('--NB', help='max num of det bbox per img', type=int, default=400)
    # parser.add_argument('--filter_', default=False, action='store_true')
    parser.add_argument('--vehicle_ensemble', default=True, action='store_true')
    # parser.add_argument('--pTest_nms_thr', help='', type=float, default=0.5)  # NOTE
    parser.add_argument('--nb_ensemble', help='', type=int, default=-1)  # NOTE
    parser.add_argument('--reweight_score', default=False, action='store_true')
    parser.add_argument('--noafter_score_filter', default=False, action='store_true')
    parser.add_argument('--nms', default='nms', type=str) # softnms,nms, matrixnms
    parser.add_argument('--sigma', default=0.5, type=float)
    parser.add_argument('--score_thresh', default=0.03, type=float)
    parser.add_argument('--method', default='linear', type=str)
    parser.add_argument('--reweight_minthr', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)

    # New for linear
    parser.add_argument('--test_mode',default=False, action='store_true')

    args = parser.parse_args()
    # config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    # return config, args
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

    # pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
    # transform, data_name, label_name, metric_list = config.get_config(is_train=False)
    # pGen = patch_config_as_nothrow(pGen)
    # pKv = patch_config_as_nothrow(pKv)
    # pRpn = patch_config_as_nothrow(pRpn)
    # pRoi = patch_config_as_nothrow(pRoi)
    # pBbox = patch_config_as_nothrow(pBbox)
    # pDataset = patch_config_as_nothrow(pDataset)
    # pModel = patch_config_as_nothrow(pModel)
    # pOpt = patch_config_as_nothrow(pOpt)
    # pTest = patch_config_as_nothrow(pTest)
    image_set_name = 'validation'  # NOTE
    
    # setting
    parent_cat_ids = {
    0: 'TYPE_VEHICLE', 
    1: 'TYPE_PEDESTRIAN', 
    2: 'TYPE_CYCLIST'}
    min_det_score = args.thresh # or pTest.min_det_score  # NOTE TODO
    print('min_det_score: {}'.format(min_det_score))
    # post process
    
    # candidate_files_val_l2map={
    #     'cascade_resnext50_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble_new': (68.89, 78.11, 60.41),
    #     'cascade_res2net50v1b_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble_new': (68.99, 78.72, 61.20),
    #     'cascade_r50v1d_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble_new': (68.78, 78.32, 59.53), 
    #     'cascade_hrfpnw32_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor'\
    #                     + '_trainpost1000_wogt_1280_2240_calibrate3stages_merge_new': (66.44, 77.84, 59.89),
    #     'cascade_hrfpnw48_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor'\
    #                     + '_trainpost1000_wogt_1280_2240_calibrate3stages_merge_new':(66.89, 78.07, 57.96),
    #     'cascade_resnext50_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor'\
    #                     + '_trainpost1000_wogt_1280_2240_calibrate3stages_merge_flip_new':(67.06, 77.54, 57.97),
    #     'cascade_res2net50_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor'\
    #                     + '_trainpost1000_wogt_calibrate3stages_merge_flip_new': (67.28, 77.83, 59.34),        
    # }
    candidate_files_val_l2map={
        ## whole model
        'submission_cascade_resnext50_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble_new': (68.89, 78.11, 60.41),
        'submission_cascade_res2net50v1b_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble_new': (68.99, 78.72, 61.20),
        'submission_cascade_r50v1d_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble_new': (68.78, 78.32, 59.53), 
        'submission_cascade_r50v1b_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble_new': (68.60, 78.38, 59.58),           # human assign
        'submission_cascade_r50v1b_hrfpnw32_pafpn6stage_1x_custom_scale_bs1_syncbn_fp16_dh4block_mst_morescale_wogt_1000proposals_4epoch_ensemble_new': (68.85, 78.08, 60.20), # human assign
        ## expert model
        'cascade_hrfpnw32_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_flip_test_calibrate3stages_merge_new': (66.44, 77.84, 59.89),
        'cascade_hrfpnw48_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_flip_test_calibrate3stages_merge_new':(66.89, 78.07, 57.96),
        'cascade_resnext50_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_test_merge_flip_new':(67.06, 77.54, 57.97),
        'cascade_res2net50_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_calibrate3stages_merge_flip_new': (67.28, 77.83, 59.34),        
    }

    
    nb_ensemble=len(candidate_files_val_l2map) if args.nb_ensemble<=0 else args.nb_ensemble
    veh_ensemble = 5; ped_ensemble = 6; cyc_ensemble = 7
    ensemble_num_list = [veh_ensemble, ped_ensemble, cyc_ensemble]
    print('nb_ensemble:{} reweight_score:{}'.format(nb_ensemble, args.reweight_score))
    assert len(candidate_files_val_l2map)>=nb_ensemble
    # val_vehicle_l2map,val_ped_l2map,val_cyc_l2map=[],[],[]
    for l2map in candidate_files_val_l2map.values():
        assert len(l2map) == 3
    #     val_vehicle_l2map.append(l2map[0])
    #     val_ped_l2map.append()
    val_vehicle_l2map_topk = OrderedDict(sorted(candidate_files_val_l2map.items(), key=lambda x: x[1][0], reverse=True)[:veh_ensemble])
    val_ped_l2map_topk = OrderedDict(sorted(candidate_files_val_l2map.items(), key=lambda x: x[1][1], reverse=True)[:ped_ensemble])
    val_cyc_l2map_topk = OrderedDict(sorted(candidate_files_val_l2map.items(), key=lambda x: x[1][2], reverse=True)[:cyc_ensemble])
    # NOTE NOTE NOTE NOTE NOTE submission_experiments
    vehicle_candidate_files=[glob.glob('submission_experiments/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_vehicle_l2map_topk.keys()]
    ped_candidate_files=[glob.glob('submission_experiments/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_ped_l2map_topk.keys()]
    cyc_candidate_files=[glob.glob('submission_experiments/{}/{}_*_before_nms.pkl'.format(filename, image_set_name)) for filename in val_cyc_l2map_topk.keys()]
    val_score_vehicle = [score[0] for score in val_vehicle_l2map_topk.values()]
    val_score_ped = [score[1] for score in val_ped_l2map_topk.values()]
    val_score_cyc = [score[2] for score in val_cyc_l2map_topk.values()]

    # NOTE reweight score
    score_list = [val_score_vehicle, val_score_ped, val_score_cyc]
    # min_reweight_thr,alpha = args.reweight_minthr, args.alpha
    alpha = args.alpha
    # def reweight_score_func()
    score_list_before = score_list
    score_list_reweighted = []

    # Single test
    if nb_ensemble != 1:
        if args.method == 'pfdet':
            for reweight_score_percls in score_list:
                max_score = np.max(reweight_score_percls)
                mean_score = np.mean(reweight_score_percls)
                print('score cmp: ', mean_score/100., args.reweight_minthr)
                min_reweight_thr = min(mean_score/100., args.reweight_minthr)
                reweight_score_percls = [(score_cur-mean_score)/(max_score-mean_score)+alpha*(max_score-score_cur)/(max_score-mean_score) if score_cur>mean_score else min_reweight_thr 
                    for score_cur in reweight_score_percls]
                minreweightscore = np.min(reweight_score_percls)
                reweight_score_percls = [minreweightscore if (score==min_reweight_thr and score>minreweightscore) else score for score in reweight_score_percls]
                assert np.max(reweight_score_percls) == 1. and reweight_score_percls[0]==1.0
                # assert np.min(reweight_score_percls) == min_reweight_thr, '{}'.format(np.min(reweight_score_percls))  # NOTE
                assert np.min(reweight_score_percls) == minreweightscore, '{}'.format(np.min(reweight_score_percls))  # NOTE
                score_list_reweighted.append(reweight_score_percls)
        elif args.method == 'linear':
            min_reweight_thr = args.reweight_minthr
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
                score_list_reweighted.append(reweight_score_percls)

        score_list = score_list_reweighted

    ### NOTE not rescore ped veh anyway
    if not args.reweight_score:
        score_list[0] = [1 for ii in range(veh_ensemble)]
        score_list[1] = [1 for ii in range(ped_ensemble)]
        score_list[2] = [1 for ii in range(cyc_ensemble)]
    print('------------------------------------------------------ ')
    print(score_list_before)
    print(score_list)
    print('score_list before and after: ------------------------- ')
    print('score_list for ensemble: ', score_list)
    # exit()
    del val_vehicle_l2map_topk, val_ped_l2map_topk, val_cyc_l2map_topk

    # assert len(vehicle_candidate_files)==len(ped_candidate_files)==len(cyc_candidate_files)
    # for i, (vehicle_file, ped_file, cyc_file) in enumerate(zip(vehicle_candidate_files, ped_candidate_files, cyc_candidate_files)):
    #     # if len(vehicle_file) == 0:
    #         # print(vehicle_file)
    #         # print(i, vehicle_candidate_files[i])
    #     #     pass
    #     assert len(vehicle_file) == len(ped_file)==len(cyc_file), '{} vs {} vs {}'.format(len(vehicle_file), len(ped_file), len(cyc_file))
    for file_ in [vehicle_candidate_files, ped_candidate_files, cyc_candidate_files]:
        for elem in zip(*file_):
            seg_id = elem[0].split('/')[-1]
            for e in elem:
                assert e.split('/')[-1]==seg_id
    vehicle_candidate_files = [elem for elem in zip(*vehicle_candidate_files)]
    ped_candidate_files = [elem for elem in zip(*ped_candidate_files)]
    cyc_candidate_files = [elem for elem in zip(*cyc_candidate_files)]
    assert len(vehicle_candidate_files) == len(ped_candidate_files) == len(cyc_candidate_files) == 149, len(vehicle_candidate_files) ## NOTE 149 for test 200 for train

    # parent_file_name = 'faster_r50v1b_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_topkanchor_mst_smallscale_zhk'
    poster='nb{}maxgt{}minscore{}'.format(nb_ensemble, args.NB, min_det_score)
    if args.reweight_score:
        if args.reweight_minthr!=0.5 or args.alpha !=0.5:
            poster += '_reweight{}_{}'.format(args.reweight_minthr, args.alpha)
        else:
            poster += '_reweight'
    if args.noafter_score_filter:
        poster += '_noafterscorefilter'
    # if args.softnms:
    if args.nms != 'nms':
        poster += '_{}_{}_{}_{}'.format(args.nms, args.sigma, args.score_thresh, args.method)
        # print('poster: ', poster)

    # save_path = os.path.join('experiments/ensemble/{}/{}_nbsemble{}'.format(image_set_name, args.save_name, nb_ensemble))
    save_path = os.path.join('nms_results/{}_{}/'.format(args.save_name, poster))
    os.makedirs(save_path, exist_ok=True)
    if args.nms == 'softnms': 
        nms_veh = cython_soft_nms_wrapper(0.6, sigma=args.sigma, score_thresh=args.score_thresh, method=args.method)
        nms_ped = cython_soft_nms_wrapper(0.5, sigma=args.sigma, score_thresh=args.score_thresh, method=args.method)
        nms_cyc = py_nms_wrapper(0.5)
        nms_list = [nms_veh, nms_ped, nms_cyc]
    elif args.nms == 'matrixnms': 
        nms = matrix_nms_wrapper(args.pTest_nms_thr, sigma=args.sigma, score_thresh=args.score_thresh, method=args.method)
    elif args.nms == 'weightnms':
        nms_veh = wnms_wrapper(0.7, 0.7)
        # nms_ped = cython_soft_nms_wrapper(0.5, sigma=args.sigma, score_thresh=args.score_thresh, method=args.method)
        nms_ped = wnms_wrapper(0.5, 0.5)
        nms_cyc = py_nms_wrapper(0.5)
        nms_list = [nms_veh, nms_ped, nms_cyc]
    else:
        nms_veh = py_nms_wrapper(0.5) 
        nms_remain = py_nms_wrapper(0.5) 
        nms_list = [nms_veh, nms_remain, nms_remain]
    # output_dict_list_allcls=[]
    for i, (elems_vehicle,elems_ped, elems_cyc) in enumerate(zip(vehicle_candidate_files, ped_candidate_files, cyc_candidate_files)):
        # assert len(elems_vehicle) ==len(elems_ped)==len(elems_cyc)== nb_ensemble
        assert len(elems_vehicle) == veh_ensemble
        assert len(elems_ped) == ped_ensemble
        assert len(elems_cyc) == cyc_ensemble
        for elem_vehicle, elem_ped, elem_cyc in zip(elems_vehicle, elems_ped, elems_cyc):
            assert elem_vehicle.split('/')[-1]==elem_cyc.split('/')[-1]==elem_cyc.split('/')[-1]
        # keys = -1
        output_dict_list = [[pkl.load(open(pkl_seg,'rb')) for pkl_seg in elems_vehicle]]
        output_dict_list.append([pkl.load(open(pkl_seg, 'rb')) for pkl_seg in elems_ped])
        output_dict_list.append([pkl.load(open(pkl_seg, 'rb')) for pkl_seg in elems_cyc])

        t1_s = time.time()

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
                nb_ensemble = ensemble_num_list[cid]
                bbox_xyxys = [output_dict_list[cid][pklsegid][k]["bbox_xyxy"] for pklsegid in range(nb_ensemble) if k in output_dict_list[cid][pklsegid]]
                cls_scores = [output_dict_list[cid][pklsegid][k]["cls_score"][:,cid] for pklsegid in range(nb_ensemble) if k in output_dict_list[cid][pklsegid]]
                # assert len(bbox_xyxys) == len(cls_scores) == nb_ensemble  # avoid none in bbox_xyxys or cls_scores
                if bbox_xyxys[0].shape[1] != 4:
                    assert bbox_xyxys[0].shape[1] == 16
                    bbox_xyxys = [bbox_xyxy[:,cid*4:(cid+1)*4] for bbox_xyxy in bbox_xyxys]

                reweight_score = [score_list[cid][cand_id] for cand_id in range(nb_ensemble)]
                # only reweight CYC
                # if cid == 2:      # NOTE modify score_list before
                #     reweight_score = [score_list[cid][cand_id] for cand_id in range(nb_ensemble)]
                # else:
                #     reweight_score = [1 for ii in range(nb_ensemble)]
                # NOTE
                # if parent_bbox_xyxy.shape[1] != 4:
                #     cls_box = parent_bbox_xyxy[:, cid * 4:(cid + 1) * 4]
                # else:
                #     cls_box = parent_bbox_xyxy

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
                nms = nms_list[cid]
                det = nms(det)
                dataest_cid = parent_cat_ids[cid]
                final_dets[dataest_cid] = det
                
            output_dict_ensembeld["det_xyxys"] = final_dets 
            del bbox_xyxys, cls_scores  # TODO check
            # del output_dict[k]["bbox_xyxy"]
            # del output_dict[k]["cls_score"]
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
            for ii in range(nb_ensemble):
                if k in output_dict_list[cid][ii]:
                    output_dict_ensembeld['meta_info']=output_dict_list[cid][ii][k]['meta_info']  # TODO check
                    break
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