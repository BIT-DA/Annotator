import os
import numpy as np
import torch
from torch.utils import data
from .semantickitti_utils import LEARNING_MAP_19, LEARNING_MAP_12, LEARNING_MAP_7, LEARNING_MAP_11
import random

# used for polarmix
instance_classes = [0, 1, 2, 3, 4, 5, 6, 7]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class SemantickittiDataset(data.Dataset):
    def __init__(
            self,
            data_cfgs=None,
            training: bool = True,
            class_names: list = None,
            root_path: str = None,
            logger=None,
            if_scribble: bool = False,
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
        self.if_scribble = if_scribble
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

        if self.split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'train_val':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10', '08']
        elif self.split == 'test':
            self.seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        else:
            raise Exception('split must be train/val/train_val/test.')

        self.annos = []
        for seq in self.seqs:
            self.annos += absoluteFilePaths('/'.join([self.root_path, str(seq).zfill(2), 'velodyne']))
        self.annos.sort()
        self.annos_another = self.annos.copy()
        random.shuffle(self.annos_another)
        print(f'The total sample is {len(self.annos)}')

        self._sample_idx = np.arange(len(self.annos))

        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.annos)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx

        # init_path = os.path.join(ABSOLUTE_PATH, 'semantickitti_init.pkl')
        self.scan_size = {}
        # if not os.path.isfile(init_path):
        for path in self.annos:
            self.scan_size[path] = np.fromfile(path, dtype=np.float32).reshape((-1, 4)).shape[0]
        # torch.save(self.scan_size, init_path)
        # else:
        #     self.scan_size = torch.load(init_path)

    def __len__(self):
        return len(self.sample_idx)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)

    def get_kitti_points_ringID(self, points):
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        yaw = -np.arctan2(scan_y, -scan_x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        ringID = np.cumsum(proj_y)
        ringID = np.clip(ringID, 0, 63)
        return ringID

    def __getitem__(self, index):
        raw_data = np.fromfile(self.annos[index], dtype=np.float32).reshape((-1, 4))

        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            if self.if_scribble:  # ScribbleKITTI (weak label)
                annos = self.annos[index].replace('SemanticKITTI', 'ScribbleKITTI')
                annotated_data = np.fromfile(
                    annos.replace('velodyne', 'scribbles')[:-3] + 'label', dtype=np.uint32
                ).reshape((-1, 1))
            else:  # SemanticKITTI (full label)
                annotated_data = np.fromfile(
                    self.annos[index].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32
                ).reshape((-1, 1))

            annotated_data = annotated_data & 0xFFFF
            if self.num_classes == 19:
                annotated_data = np.vectorize(LEARNING_MAP_19.__getitem__)(annotated_data)
            elif self.num_classes == 12:
                annotated_data = np.vectorize(LEARNING_MAP_12.__getitem__)(annotated_data)
            elif self.num_classes == 7:
                annotated_data = np.vectorize(LEARNING_MAP_7.__getitem__)(annotated_data)
            elif self.num_classes == 11:
                annotated_data = np.vectorize(LEARNING_MAP_11.__getitem__)(annotated_data)
            else:
                raise NotImplementedError

        ringID = self.get_kitti_points_ringID(raw_data).reshape((-1, 1))
        raw_data = np.concatenate([raw_data, ringID.reshape(-1, 1)], axis=1).astype(np.float32)
        pc_data = {
            'xyzret': raw_data,
            'labels': annotated_data.astype(np.uint8),
            'path': self.annos[index],
        }

        return pc_data

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError
