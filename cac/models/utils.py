from os.path import join, basename
from glob import glob
from natsort import natsorted
import torch
import math
import numpy as np
from cac.data.utils import read_dataset_from_config


def get_saved_checkpoint_path(load_dir: str, load_best: bool = False, epoch: int = -1) -> str:
    """Returns the filename of saved checkpoint in given folder and for given epoch.

    :param load_dir: folder containing the required saved checkpoint
    :type load_dir: str
    :param load_best: whether to load best checkpoint or not
    :type load_best: bool
    :param epoch: epoch on which the checkpoint was saved
    :type epoch: int

    :return: path to the required checkpoint
    """
    if load_best:
        fname = 'best_ckpt.pth.tar'
    else:
        available_ckpts = natsorted(glob(join(load_dir, '[0-9]*_ckpt.pth.tar')))
        if epoch == -1:
            fname = basename(available_ckpts[-1])
        else:
            fname = '{}_ckpt.pth.tar'.format(epoch)

    return join(load_dir, fname)


def get_subsets(subset_tracker_config):
    mode_subsets = dict()
    for subset_config in subset_tracker_config:
        # each subset has its own data config with a corresponding
        # `mode` and we keep a dictionary of subset `mode` and the
        # corresponding IDs
        subset_info = read_dataset_from_config(subset_config)

        # converting to set as comparison becomes faster than a list
        mode_subsets[subset_config['mode']] = set(
            subset_info['file'])

    return mode_subsets


def tensorize(x):
    """Tensorize a np.ndarray"""
    if not isinstance(x, torch.Tensor):
        return torch.Tensor(x)
    return x


def logit(z: np.float32) -> np.float32:
    """Returns the logit (inverse of sigmoid)

    :param z: scaler value
    :type z: np.float32

    :returns: logit(z)
    """
    epsilon = np.finfo(np.float32).eps
    return np.log((z / ((1 - z) + epsilon)) + epsilon)
