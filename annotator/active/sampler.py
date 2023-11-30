import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import torch_scatter
import torchsparse
import torchsparse.nn.functional

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def torch_unique(x):
    unique, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inds = torch_scatter.scatter_min(perm, inverse, dim=0)[0]
    return unique, inds, inverse, counts


def get_point_in_voxel(raw_coord, voxel_size=0.25, max_points_per_voxel=100, filter_ratio=0.01):
    voxel_grid = (raw_coord / voxel_size).int()
    hash_tensor = torch.cat((voxel_grid, torch.zeros((voxel_grid.shape[0], 1), device=voxel_grid.device)), dim=1).int()
    pc_hash = torchsparse.nn.functional.sphash(hash_tensor)
    sparse_hash, voxel_idx, inverse, voxel_point_counts = torch_unique(pc_hash)
    voxelized_coordinates = voxel_grid[voxel_idx]

    inverse_sorted, sorted_to_inverse_index = torch.sort(inverse)
    index = torch.arange(inverse_sorted.shape[0], device=raw_coord.device)
    first_locate_index = torch_scatter.scatter_min(index, inverse_sorted, dim=0)[0]

    k = int(voxelized_coordinates.shape[0] * filter_ratio)
    max_voxel_point = min(torch.topk(voxel_point_counts, k=k, largest=True)[0][-1], max_points_per_voxel)

    temp = first_locate_index[:, None] - torch.zeros(max_voxel_point, device=raw_coord.device, dtype=torch.int64)
    sorted_locate_index = temp + torch.arange(0, max_voxel_point, device=raw_coord.device)
    point_index_bound = torch.cat((temp, torch.full([1, max_voxel_point], raw_coord.shape[0], device=raw_coord.device)),
                                  dim=0)[1:]
    sorted_to_inverse_index_temp = torch.cat((sorted_to_inverse_index, torch.tensor([-1], device=raw_coord.device)),
                                             dim=0)
    point_in_voxel = sorted_to_inverse_index_temp[
        torch.where(point_index_bound > sorted_locate_index, sorted_locate_index, -1)]

    point_in_voxel[inverse[point_in_voxel[:, 0]]] = point_in_voxel.clone()

    voxel_point_counts = torch.clamp(voxel_point_counts, 0, max_voxel_point)
    return voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts


class RandomSelect:
    def __init__(self,
                 select_num: int = 1,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100, ):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel

    def select(self, mask, raw_coord=None, preds=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)
            perm = torch.randperm(voxel_idx.shape[0])
            index = perm[mask[point_in_voxel[perm][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
        else:
            raise NotImplementedError


class EntropySelect:
    def __init__(self,
                 select_num: int = 1,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.voxel_select_method = voxel_select_method

    def select(self, mask, raw_coord=None, preds=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            conf = F.softmax(preds, dim=-1)
            log2_conf = torch.log2(conf)
            entropy = -torch.mul(conf, log2_conf).sum(dim=1)
            entropy[-1] = 0

            point_entropy_in_voxel = entropy[point_in_voxel]
            if self.voxel_select_method == 'max':
                voxel_entropy = torch.max(point_entropy_in_voxel, dim=1)[0]
            elif self.voxel_select_method == 'mean':
                voxel_entropy = torch.sum(point_entropy_in_voxel, dim=1)
                voxel_entropy = voxel_entropy / voxel_point_counts
            else:
                raise NotImplementedError

            _, voxel_indices = torch.sort(voxel_entropy, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
        else:
            raise NotImplementedError


class MarginSelect:
    def __init__(self,
                 select_num: int = 1,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.voxel_select_method = voxel_select_method

    def select(self, mask, raw_coord=None, preds=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            conf = F.softmax(preds, dim=-1)
            top2_conf, _ = torch.topk(conf, k=2, largest=True, dim=1, sorted=True)
            sub_result = top2_conf[:, 0] - top2_conf[:, 1]

            point_conf_in_voxel = sub_result[point_in_voxel]
            if self.voxel_select_method == 'max':
                voxel_conf = torch.max(point_conf_in_voxel, dim=1)[0]
            elif self.voxel_select_method == 'mean':
                voxel_conf = torch.sum(point_conf_in_voxel, dim=1)
                voxel_conf = voxel_conf / voxel_point_counts
            else:
                raise NotImplementedError

            _, voxel_indices = torch.sort(voxel_conf, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
        else:
            raise NotImplementedError


class VCDSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
        else:
            raise NotImplementedError
