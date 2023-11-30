import os
import numpy as np
import torch
from torch.utils import data
from .synlidar_utils import LEARNING_MAP_KITTI, LEARNING_MAP_nuscenes, LEARNING_MAP_7, LEARNING_MAP_13
import random

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class SynlidarDataset(data.Dataset):
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
        self.target = data_cfgs.TARGET
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

        self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        self.splits = {}
        self.get_splits()

        self.annos = []
        for sequence, frames in self.splits[self.split].items():
            for frame in frames:
                pcd_path = os.path.join(self.root_path, sequence, 'velodyne', f'{int(frame):06d}.bin')
                self.annos.append(pcd_path)
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

    def get_splits(self):
        split_path = os.path.join(ABSOLUTE_PATH, 'synlidar_split.pkl')

        if not os.path.isfile(split_path):
            self.splits = {'train': {s: [] for s in self.sequences},
                           'val': {s: [] for s in self.sequences}}
            for sequence in self.sequences:
                data_path = os.listdir(os.path.join(self.root_path, sequence, 'labels'))
                num_frames = len(data_path)
                valid_frames = []
                data_index = []
                for label_name in data_path:
                    data_index.append(int(label_name.split('.')[0]))
                for v in data_index:
                    pcd_path = os.path.join(self.root_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                    label_path = os.path.join(self.root_path, sequence, 'labels', f'{int(v):06d}.label')

                    if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                        valid_frames.append(v)

                validation_selected = np.random.choice(valid_frames, int(num_frames * 0.1), replace=False)
                for t in validation_selected:
                    valid_frames.remove(t)

                train_selected = valid_frames

                self.splits['train'][sequence].extend(train_selected)
                self.splits['val'][sequence].extend(validation_selected)
            torch.save(self.splits, split_path)

        else:
            self.splits = torch.load(split_path)
            print('SEQUENCES', self.splits.keys())
            print('TRAIN SEQUENCES', self.splits['train'].keys())

    def __len__(self):
        return len(self.sample_idx)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.annos[index], dtype=np.float32).reshape((-1, 4))

        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(
                self.annos[index].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32
            ).reshape((-1, 1))

            annotated_data = annotated_data & 0xFFFF
            if self.target == 'kitti':
                if self.num_classes == 7:
                    annotated_data = np.vectorize(LEARNING_MAP_7.__getitem__)(annotated_data)
                else:
                    annotated_data = np.vectorize(LEARNING_MAP_KITTI.__getitem__)(annotated_data)
            elif self.target == 'nuscenes':
                if self.num_classes == 7:
                    annotated_data = np.vectorize(LEARNING_MAP_7.__getitem__)(annotated_data)
                else:
                    annotated_data = np.vectorize(LEARNING_MAP_nuscenes.__getitem__)(annotated_data)
            elif self.target == 'poss':
                if self.num_classes == 13:
                    annotated_data = np.vectorize(LEARNING_MAP_13.__getitem__)(annotated_data)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        pc_data = {
            'xyzret': raw_data,
            'labels': annotated_data.astype(np.uint8),
            'path': self.annos[index],
        }

        return pc_data

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError
