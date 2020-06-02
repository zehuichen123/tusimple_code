import numpy as np
import pickle as pkl
import os
from glob import glob
import mmcv

fold_list = [
    # # 'cascade_r50v1d_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_smallscale_4epoch_wogt_1000proposal_test',
    # # 'cascade_r50v1b_hrfpnw32_pafpn6stage_1x_custom_scale_bs1_syncbn_fp16_dh4block_mst_smallscale_3all_test',
    # # 'cascade_hrfpnw32_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_1600'
    # # 'cascade_hrfpnw32_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_' \
    # #                                 + 'topkanchor_trainpost1000_wogt_1280_2240_calibrate3stages_merge',
    # # 'cascade_hrfpnw48_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_' \
    # #                                 + 'topkanchor_trainpost1000_wogt_1280_2240_calibrate3stages_merge',
    # # 'cascade_resnext50_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_'\
    # #                         + 'sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_1600_calibrate3stages_merge'
    # # 'cascade_resnext50_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1'\
    # #                         + '_bg10000_e10_topkanchor_trainpost1000_wogt_1280_2240_calibrate3stages_merge_flip',
    # # 'cascade_res2net50_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_'\
    # #                         + 'repeadload_sizethr-1_bg10000_e10_topkanchor_trainpost1000_wogt_calibrate3stages_merge',
    # # 'cascade_hrfpnw18_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10'\
    # #                         + '_topkanchor_1280_2240_calibrate3stages_merge_flip',
    # 'cascade_res2net50_pafpn_1x_custom_scale_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10'\
    #                         + '_topkanchor_trainpost1000_wogt_calibrate3stages_merge_flip'
    # # 'cascade_r50v1d_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble',
    # # 'cascade_r50v1b_hrfpnw32_pafpn6stage_1x_custom_scale_bs1_syncbn_fp16_dh4block_mst_smallscale_ensemble'
    # 'cascade_hrfpnw32_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_sizethr-1_bg10000_e10_topkanchor'\
    #                         + '_trainpost1000_wogt_flip_test_calibrate3stages_merge',
    # 'cascade_hrfpnw48_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt'\
    #                             + '_flip_test_calibrate3stages_merge',
    # 'cascade_res2net50_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_calibrate3stages_merge_flip',
    # 'cascade_resnext50_pafpn_syncbn_fp16_iou506070_mst_morescale_addvehicle_repeadload_e10_topkanchor_trainpost1000_wogt_test_merge_flip',
    # 'submission_cascade_r50v1b_hrfpnw32_pafpn6stage_1x_custom_scale_bs1_syncbn_fp16_dh4block_mst_morescale_wogt_1000proposals_4epoch_ensemble',
    # 'submission_cascade_r50v1b_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble',
    # 'submission_cascade_r50v1d_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble'
    # 'submission_cascade_res2net50v1b_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble',
    'submission_cascade_resnext50_pafpn3stage_1x_custom_scale_bs2_syncbn_fp16_dh3block_mst_morescale_4epoch_wogt_1000proposal_ensemble'
]

def get_rank(filename):
    filename = filename.split('/')[-1]
    res_list = filename.split('_')
    if 'rank' not in filename:
        rank = 0; index = int(res_list[1])
    else:
        rank = int(res_list[1][-1]); index = int(res_list[2])
    return rank * 100000 + index

file1_list = glob('submission_experiments/%s/testing*.pkl' % fold_list[0])
res_list = []

for file in file1_list:
    data = pkl.load(open(file, 'rb'), encoding='latin1')
    res_list += list(data.items())

res_list = sorted(res_list, key=lambda x: x[0])

class DataPaser():
    def __init__(self, fold_name):
        self.all_cnt = 0
        self.cnt = 0
        self.res_list = []
        self.fold_name = fold_name
        os.makedirs(self.fold_name, exist_ok=True)

    def add(self, data):
        self.res_list.append(data)
        self.cnt += 1
        if self.cnt == 1000:
            self.dump()

    def dump(self):
        if self.cnt == 0:
            return
        start_num = self.all_cnt * 1000; end_num = (self.all_cnt + 1) * 1000
        save_path = self.fold_name + 'validation_%d_%d_before_nms.pkl' % (start_num, end_num)
        with open(save_path, "wb") as fout:
            pkl.dump(dict(self.res_list), fout)
        print("Dump to %s" % save_path)
        self.cnt = 0
        self.all_cnt += 1
        self.res_list = []

data = DataPaser('submission_experiments/%s_new/'%fold_list[0])
for res in res_list:
    data.add(res)
data.dump()
print("Parsing Line %d" % data.all_cnt)


