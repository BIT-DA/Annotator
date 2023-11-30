'''
This file is modified from https://github.com/mit-han-lab/spvnas
'''
from collections import defaultdict

import numba as nb
import numpy as np
import torch
from torch.utils import data
from .synlidar import SynlidarDataset
from tools.utils.common.seg_utils import aug_points

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 0], input_xyz[:, 1])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

class SynlidarBevDataset(data.Dataset):
    def __init__(
            self,
            data_cfgs=None,
            training=True,
            root_path=None,
            logger=None,
    ):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.training = training
        self.target = data_cfgs.TARGET
        self.num_classes = data_cfgs.NUM_CLASSES
        if self.target == 'kitti':
            if self.num_classes == 7:
                self.class_names = [  # 7
                    "unlabeled",  # ignored
                    "vehicle", "pedestrian", "road", "sidewalk", "terrain", "manmade",
                    "vegetation",
                ]
            else:
                self.class_names = [  # 19
                    "unlabeled",  # ignored
                    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist",
                    # dynamic
                    "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk",
                    "terrain",
                    "pole", "traffic-sign"  # static
                ]
        elif self.target == 'nuscenes':
            if self.num_classes == 7:
                self.class_names = [  # 7
                    "unlabeled",  # ignored
                    "vehicle", "pedestrian", "road", "sidewalk", "terrain", "manmade",
                    "vegetation",
                ]
            else:
                self.class_names = [  # 13
                    "unlabeled",  # ignored
                    "car", "bicycle", "motorcycle", "truck", "bus", "person",
                    "road", "sidewalk", "other-ground", "vegetation", "terrain", "manmade",
                    "traffic-cone"
                ]
        elif self.target == 'poss':
            if self.num_classes == 13:
                self.class_names = [  # 13
                    "unlabeled",  # ignored
                    "car", "bicycle", "person", "rider", "ground", "building", "fence", "plants",
                    "trunk", "pole", "traffic-sign", "garbage-can", "cone/stone"
                ]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.root_path = root_path if root_path is not None else self.data_cfgs.DATA_PATH
        self.logger = logger

        self.point_cloud_dataset = SynlidarDataset(
            data_cfgs=data_cfgs,
            training=training,
            class_names=self.class_names,
            root_path=self.root_path,
            logger=logger,
        )

        self.num_points = data_cfgs.NUM_POINTS

        self.if_flip = data_cfgs.get('FLIP_AUG', True)
        self.if_scale = data_cfgs.get('SCALE_AUG', True)
        self.scale_axis = data_cfgs.get('SCALE_AUG_AXIS', 'xyz')
        self.scale_range = data_cfgs.get('SCALE_AUG_RANGE', [0.9, 1.1])
        self.if_jitter = data_cfgs.get('TRANSFORM_AUG', True)
        self.if_rotate = data_cfgs.get('ROTATE_AUG', True)

        self.if_tta = self.data_cfgs.get('TTA', False)

        self.grid_size = np.asarray(data_cfgs.get('GRID_SIZE', [480, 360, 32]))
        self.ignore_label = data_cfgs.IGNORE_LABEL

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        if self.if_tta:
            data_total = []
            voting = 10
            for idx in range(voting):
                data_single = self.get_single_sample(index, idx)
                data_total.append(data_single)
            return data_total
        else:
            data = self.get_single_sample(index)
            return data

    def get_single_sample(self, index, voting_idx=0):
        'Generates one sample of data'
        pc_data = self.point_cloud_dataset[index]
        labels = pc_data['labels'].reshape(-1)
        xyz = pc_data['xyzret'][:, :4].astype(np.float32)

        raw_coord = xyz.copy()[:, :3]
        num_points_current_frame = xyz.shape[0]
        if self.training:
            xyz[:, 0:3] = aug_points(
                xyz=xyz[:, :3],
                if_flip=self.if_flip,
                if_scale=self.if_scale,
                scale_axis=self.scale_axis,
                scale_range=self.scale_range,
                if_jitter=self.if_jitter,
                if_rotate=self.if_rotate,
                if_tta=self.if_tta,
            )

        elif self.if_tta:
            self.if_flip = False
            self.if_scale = True
            self.scale_aug_range = [0.95, 1.05]
            self.if_jitter = False
            self.if_rotate = True
            xyz[:, 0:3] = aug_points(
                xyz=xyz[:, :3],
                if_flip=self.if_flip,
                if_scale=self.if_scale,
                scale_axis=self.scale_axis,
                scale_range=self.scale_range,
                if_jitter=self.if_jitter,
                if_rotate=self.if_rotate,
                if_tta=True,
                num_vote=voting_idx,
            )

        other_feature = xyz[:, 3:]

        xyz_pol = cart2polar(xyz)

        max_volume_space = [50, np.pi, 1.5]
        min_volume_space = [3, -np.pi, -3]
        max_bound = np.asarray(max_volume_space)
        min_bound = np.asarray(min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels.reshape((-1, 1))], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        valid_label = np.zeros_like(processed_label, dtype=bool)
        valid_label[grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label, axis=0)
        max_distance = max_bound[0] - intervals[0] * (max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2) - np.transpose(voxel_position[0], (1, 2, 0))
        distance_feature = np.transpose(distance_feature, (1, 2, 0))
        # convert to boolean feature
        distance_feature = (distance_feature > 0) * -1.
        distance_feature[grid_ind[:, 2], grid_ind[:, 0], grid_ind[:, 1]] = 1.

        data_dict = {'distance_feature': distance_feature,
                     'processed_label': processed_label,
                     'name': pc_data['path'],
                     'raw_coord': raw_coord,
                     'raw_point': xyz, }

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        return_fea = np.concatenate((return_xyz, other_feature), axis=1)

        data_dict['grid_ind'] = grid_ind
        data_dict['labels'] = labels
        data_dict['return_fea'] = return_fea
        data_dict['index'] = index

        return data_dict

    @staticmethod
    def collate_batch(inputs):
        data_dict = defaultdict(list)
        for cur_sample in inputs:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        ret = {}

        for key, val in data_dict.items():
            try:
                # seg
                if key in ['distance_feature']:
                    ret[key] = np.stack(val).astype(np.float32)
                elif key in ['processed_label']:
                    ret[key] = np.stack(val)
                elif key in ['name', 'grid_ind', 'labels', 'preseg_labels', 'return_fea', 'index', 'raw_point', 'raw_coord', 'valid_index']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        return ret

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label