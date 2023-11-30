#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter
from annotator.loss import Losses

from .bev_unet import BEV_Unet


class Polarnet(nn.Module):

    def __init__(self,
                 model_cfgs,
                 num_class,):
        super(Polarnet, self).__init__()
        self.model_cfg = model_cfgs
        self.num_class = num_class

        self.grid_size = model_cfgs.get('GRID_SIZE', [480, 360, 32])
        self.ignore_label = model_cfgs.get('IGNORE_LABEL', 0)
        self.pt_model = model_cfgs.get('PT_MODEL', 'pointnet')
        self.pt_pooling = model_cfgs.get('PT_POOLING', 'max')
        self.max_pt = model_cfgs.get('MAX_PT_PER_ENCODE', 64)
        self.pt_selection = model_cfgs.get('PT_SELECTION', 'random')
        self.fea_compre = model_cfgs.get('FEA_COMPRE', None)
        self.fea_dim = model_cfgs.get('FEA_DIM', 9)
        self.out_pt_fea_dim = model_cfgs.get('OUT_PT_FEA_DIM', 64)
        self.kernal_size = model_cfgs.get('KERNAL_SIZE', 3)

        self.BEV_model = BEV_Unet(n_class=self.num_class,
                                  n_height=self.grid_size[2],
                                  input_batch_norm=True,
                                  dropout=0.5,
                                  circular_padding=True)
        self.global_step = 0

        assert self.pt_pooling in ['max']
        assert self.pt_selection in ['random', 'farthest']

        if self.pt_model == 'pointnet':
            self.PPmodel = nn.Sequential(
                nn.BatchNorm1d(self.fea_dim),

                nn.Linear(self.fea_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),

                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),

                nn.Linear(256, self.out_pt_fea_dim)
            )

        # NN stuff
        if self.kernal_size != 1:
            if self.pt_pooling == 'max':
                self.local_pool_op = torch.nn.MaxPool2d(self.kernal_size, stride=1, padding=(self.kernal_size - 1) // 2,
                                                        dilation=1)
            else:
                raise NotImplementedError
        else:
            self.local_pool_op = None

        # parametric pooling        
        if self.pt_pooling == 'max':
            self.pool_dim = self.out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

        label_smoothing = model_cfgs.get('LABEL_SMOOTHING', 0.0)
        default_loss_config = {
            'LOSS_TYPES': ['CELoss', 'LovLoss'],
            'LOSS_WEIGHTS': [1.0, 1.0],
        }
        loss_config = self.model_cfg.get('LOSS_CONFIG', default_loss_config)

        loss_types = loss_config.get('LOSS_TYPES', default_loss_config['LOSS_TYPES'])
        loss_weights = loss_config.get('LOSS_WEIGHTS', default_loss_config['LOSS_WEIGHTS'])
        assert len(loss_types) == len(loss_weights)

        self.criterion_losses = Losses(
            loss_types=loss_types,
            loss_weights=loss_weights,
            ignore_index=self.ignore_label,
            label_smoothing=label_smoothing,
        )

    def update_global_step(self):
        self.global_step += 1

    def forward(self, batch_dict, voxel_fea=None):
        pt_fea = batch_dict['return_fea']
        xy_ind = [i[:, :2] for i in batch_dict['grid_ind']]
        cur_dev = pt_fea[0].get_device()

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # subsample pts
        if self.pt_selection == 'random':
            grp_ind = grp_range_torch(unq_cnt, cur_dev)[torch.argsort(torch.argsort(unq_inv))]
            remain_ind = grp_ind < self.max_pt
        elif self.pt_selection == 'farthest':
            unq_ind = np.split(np.argsort(unq_inv.detach().cpu().numpy()),
                               np.cumsum(unq_cnt.detach().cpu().numpy()[:-1]))
            remain_ind = np.zeros((pt_num,), dtype=np.bool)
            np_cat_fea = cat_pt_fea.detach().cpu().numpy()[:, :3]
            pool_in = []
            for i_inds in unq_ind:
                if len(i_inds) > self.max_pt:
                    pool_in.append((np_cat_fea[i_inds, :], self.max_pt))
            if len(pool_in) > 0:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                FPS_results = pool.starmap(parallel_FPS, pool_in)
                pool.close()
                pool.join()
            count = 0
            for i_inds in unq_ind:
                if len(i_inds) <= self.max_pt:
                    remain_ind[i_inds] = True
                else:
                    remain_ind[i_inds[FPS_results[count]]] = True
                    count += 1

        cat_pt_fea = cat_pt_fea[remain_ind, :].float()
        cat_pt_ind = cat_pt_ind[remain_ind, :]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt, max=self.max_pt)

        # process feature
        if self.pt_model == 'pointnet':
            processed_cat_pt_fea = self.PPmodel(cat_pt_fea)

        if self.pt_pooling == 'max':
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
        else:
            raise NotImplementedError

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        # stuff pooled data into 4D tensor
        out_data_dim = [len(pt_fea), self.grid_size[0], self.grid_size[1], self.pt_fea_dim]
        out_data = torch.zeros(out_data_dim, dtype=processed_pooled_data.dtype).to(cur_dev)
        out_data[unq[:, 0], unq[:, 1], unq[:, 2], :] = processed_pooled_data
        out_data = out_data.permute(0, 3, 1, 2)
        if self.local_pool_op != None:
            out_data = self.local_pool_op(out_data)
        if voxel_fea is not None:
            out_data = torch.cat((out_data, voxel_fea), 1)

        # run through network
        out = self.BEV_model(out_data)

        return {'network_loss': self.criterion_losses,
                'predict_logits': out, }

def grp_range_torch(a, dev):
    idx = torch.cumsum(a, 0)
    id_arr = torch.ones(idx[-1], dtype=torch.int64, device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    return torch.cumsum(id_arr, 0)


def parallel_FPS(np_cat_fea, K):
    return nb_greedy_FPS(np_cat_fea, K)


@nb.jit('b1[:](f4[:,:],i4)', nopython=True, cache=True)
def nb_greedy_FPS(xyz, K):
    start_element = 0
    sample_num = xyz.shape[0]
    sum_vec = np.zeros((sample_num, 1), dtype=np.float32)
    xyz_sq = xyz ** 2
    for j in range(sample_num):
        sum_vec[j, 0] = np.sum(xyz_sq[j, :])
    pairwise_distance = sum_vec + np.transpose(sum_vec) - 2 * np.dot(xyz, np.transpose(xyz))

    candidates_ind = np.zeros((sample_num,), dtype=np.bool_)
    candidates_ind[start_element] = True
    remain_ind = np.ones((sample_num,), dtype=np.bool_)
    remain_ind[start_element] = False
    all_ind = np.arange(sample_num)

    for i in range(1, K):
        if i == 1:
            min_remain_pt_dis = pairwise_distance[:, start_element]
            min_remain_pt_dis = min_remain_pt_dis[remain_ind]
        else:
            cur_dis = pairwise_distance[remain_ind, :]
            cur_dis = cur_dis[:, candidates_ind]
            min_remain_pt_dis = np.zeros((cur_dis.shape[0],), dtype=np.float32)
            for j in range(cur_dis.shape[0]):
                min_remain_pt_dis[j] = np.min(cur_dis[j, :])
        next_ind_in_remain = np.argmax(min_remain_pt_dis)
        next_ind = all_ind[remain_ind][next_ind_in_remain]
        candidates_ind[next_ind] = True
        remain_ind[next_ind] = False

    return candidates_ind
