
from .voxel.minknet.minknet import MinkNet
from .voxel.spvcnn.spvcnn import SPVCNN

__all__ = {
    'MinkNet': MinkNet,
    'SPVCNN': SPVCNN,
}


def build_segmentor(model_cfgs, num_class):
    model = eval(model_cfgs.NAME)(
        model_cfgs=model_cfgs,
        num_class=num_class,
    )

    return model
