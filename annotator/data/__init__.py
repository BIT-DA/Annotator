import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from tools.utils.common import common_utils

from .dataset.synlidar import SynlidarDataset, SynlidarVoxelDataset
from .dataset.semantickitti import SemkittiVoxelDataset
from .dataset.semanticposs import SempossVoxelDataset
from .dataset.nuscenes import NuscDataset, NuscVoxelDataset

__all__ = {

    # Synlidar
    'SynlidarDataset': SynlidarDataset,
    'SynlidarVoxelDataset': SynlidarVoxelDataset,

    # SemanticKITTI
    'SemkittiVoxelDataset': SemkittiVoxelDataset,

    # SemanticPOSS
    'SempossVoxelDataset': SempossVoxelDataset,

    # nuScenes
    'NuscDataset': NuscDataset,
    'NuscVoxelDataset': NuscVoxelDataset,
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(
        data_cfgs,
        modality: str,
        batch_size: int,
        dist: bool = False,
        workers: int = 10,
        logger=None,
        training: bool = True,
        merge_all_iters_to_one_epoch: bool = False,
        total_epochs: int = 0,
):
    assert modality == 'voxel'

    dataset = data_cfgs.DATASET if training else data_cfgs.TARGET
    root_path = data_cfgs.DATA_PATH if training else data_cfgs.TARGET_PATH

    if dataset == 'nuscenes' or dataset == 'nus':
        db = 'NuscVoxelDataset'
    elif dataset == 'semantickitti' or dataset == 'kitti':
        db = 'SemkittiVoxelDataset'
    elif dataset == 'semanticposs' or dataset == 'poss':
        db = 'SempossVoxelDataset'
    elif dataset == 'synlidar':
        db = 'SynlidarVoxelDataset'
    else:
        raise NotImplementedError

    dataset = eval(db)(
        data_cfgs=data_cfgs,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    tta = data_cfgs.get('TTA', False)
    if tta:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=workers,
            shuffle=(sampler is None) and training,
            collate_fn=dataset.collate_batch_tta,
            drop_last=False,
            sampler=sampler,
            timeout=0,
            persistent_workers=(workers > 0),
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=workers,
            shuffle=(sampler is None) and training,
            collate_fn=dataset.collate_batch,
            drop_last=False,
            sampler=sampler,
            timeout=0,
            persistent_workers=(workers > 0),
        )
    return dataset, dataloader, sampler
