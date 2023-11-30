import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils import data
from nuscenes.nuscenes import NuScenes as nuScenes
from .nuscenes_utils import NuscenesLidarMethods
from .nuscenes_utils import LEARNING_MAP_12, LEARNING_MAP_13, LEARNING_MAP_7
import random

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class NuscDataset(data.Dataset):
    def __init__(
            self,
            data_cfgs=None,
            training: bool = True,
            class_names: list = None,
            root_path: str = None,
            logger=None,
    ):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.root_path = root_path
        self.training = training
        self.logger = logger
        self.class_names = class_names
        self.tta = data_cfgs.get('TTA', False)
        self.train_val = data_cfgs.get('TRAINVAL', False)
        self.augment = data_cfgs.AUGMENT
        self.num_classes = data_cfgs.NUM_CLASSES

        if self.training and not self.train_val:
            self.split = 'train'
        else:
            if self.training and self.train_val:
                self.split = 'train_val'
            else:
                self.split = 'val'
        if self.tta:
            self.split = 'test'

        self.db = nuScenes('v1.0-trainval', root_path, True, 0.1)
        self._tokens = self.tokens

        self._sample_idx = np.arange(len(self._tokens))

        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self._tokens)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx

        # init_path = os.path.join(ABSOLUTE_PATH, 'nuscenes_init.pkl')
        self.scan_size = {}
        # if not os.path.isfile(init_path):
        for token in self._tokens:
            sample = self.db.get('sample', token)
            sample_data_token = sample['data']['LIDAR_TOP']
            sd_rec = self.db.get('sample_data', sample_data_token)
            scan = np.fromfile(os.path.join(self.db.dataroot, sd_rec['filename']), dtype=np.float32)
            self.scan_size[token] = scan.reshape((-1, 5)).shape[0]
        #     torch.save(self.scan_size, init_path)
        # else:
        #     self.scan_size = torch.load(init_path)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)

    @property
    def db_version(self) -> str:
        return self.db.version

    @property
    def tokens(self) -> List[str]:
        sample_tokens = NuscenesLidarMethods.splits(split=self.split, db=self.db)
        return sample_tokens

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        token = self._tokens[index]

        sample = self.db.get('sample', token)
        sample_data_token = sample['data']['LIDAR_TOP']
        sd_rec = self.db.get('sample_data', sample_data_token)

        scan = np.fromfile(os.path.join(self.db.dataroot, sd_rec['filename']), dtype=np.float32)
        raw_data = scan.reshape((-1, 5))

        lidarseg_labels_filename = os.path.join(
            self.db.dataroot,
            self.db.get('lidarseg', sample_data_token)['filename']
        )
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

        assert len(points_label) == len(raw_data), \
            f'lidar seg labels {len(points_label)} does not match lidar points {len(raw_data)}'

        if self.num_classes == 13:
            points_label = np.vectorize(LEARNING_MAP_13.__getitem__)(points_label)
        elif self.num_classes == 12:
            points_label = np.vectorize(LEARNING_MAP_12.__getitem__)(points_label)
        elif self.num_classes == 7:
            points_label = np.vectorize(LEARNING_MAP_7.__getitem__)(points_label)
        else:
            raise NotImplementedError

        points_label = np.array(points_label)

        pc_data = {
            'xyzret': raw_data,
            'labels': points_label.astype(np.uint8),
            'path': self._tokens[index],
        }

        return pc_data

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError
