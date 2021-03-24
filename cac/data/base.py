"""Defines base dataset class for loading audio datasets."""
import random
from os.path import join
from typing import Tuple, List
from glob import glob

import numpy as np
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import Dataset

from cac.data.audio import AudioItem
from cac.data.utils import read_dataset_from_config
from cac.utils.typing import DatasetConfigDict


class BaseDataset(Dataset):
    """
    Defines the base dataset object that needs to be inherited
    by any task-specific dataset class.

    If `as_frames=True`, each signal is divided into fixed-length frames. If the
    signal is shorter than the `frame_length`, the signal is simply returned as a
    frame. When signal is longer than `frame_length`, the last frame is most likely
    to not be equal to the `frame_length`. So, the last `frame_length` milliseconds
    make the final frame.

    :param dataset_config: defines the config for the data to be loaded. The config is
        specified by a list of dicts, with each dict representing
        (dataset_name, dataset_version, mode [train, test, val])
    :type dataset_config: DatasetConfigDict
    :param fraction: fraction of the data to load, defaults to 1.0
    :type fraction: float
    :param as_frames: whether to split each file into fixed-length frames,
        defaults to False
    :type as_frames: bool
    :param frame_length: length of each fixed frame in milliseconds; used only
        when as_frames=True, defaults to None
    :type frame_length: int
    :param hop_length: number of milliseconds to skip between consecutive
        frames, defaults to 500
    :type hop_length: int
    :param min_length: minimum length of the final frame for it to be considered
        a valid frame, defaults to 100
    :type min_length: int
    """
    def __init__(self, dataset_config: List[DatasetConfigDict],
                 fraction: float = 1.0, as_frames: bool = False,
                 frame_length: int = None, hop_length: int = 500,
                 min_length: int = 100):
        self._check_args(fraction, as_frames, frame_length)
        self.dataset_config = dataset_config
        self.as_frames = as_frames

        if as_frames:
            # converting from milliseconds to seconds
            self.frame_length = frame_length / 1000
            self.hop_length = hop_length / 1000
            self.min_length = min_length / 1000

        self.load_datasets()
        self.load_fraction(fraction)

    def load_datasets(self):
        """Load the dataset as specified by self.dataset_config"""
        self.items = []

        for dataset_config in self.dataset_config:
            data_info = read_dataset_from_config(dataset_config)

            for i in tqdm(range(len(data_info['file'])), desc='Loading items'):
                path, label = data_info['file'][i], data_info['label'][i]

                start = data_info['start'][i] if 'start' in data_info.keys() else None
                end = data_info['end'][i] if 'end' in data_info.keys() else None

                if self.as_frames:
                    audio_items = self._create_frames(path, label, start, end)
                    self.items.extend(audio_items)

                else:
                    audio_item = AudioItem(path=path, label=label, start=start, end=end)
                    self.items.append(audio_item)

    def _create_frames(self, path, label, start, end):
        if start is None or end is None:
            raise ValueError('start and end values cannot be None for as_frames=True')

        # if the file is smaller than frame_length, simply return one audio item
        if end - start < self.frame_length:
            return [AudioItem(path=path, label=label, start=start, end=end)]

        steps = np.arange(start, end, self.hop_length)
        items = []

        for step in steps:
            # this indicates the last frame
            if end - step < self.frame_length:
                # check if it is long enough to be considered
                if end - (step + self.hop_length) > self.min_length:
                    _start = end - self.frame_length
                    items.append(AudioItem(path=path, label=label, start=_start, end=end))
                break

            _end = step + self.frame_length
            items.append(AudioItem(path=path, label=label, start=step, end=_end))

        return items

    def load_fraction(self, fraction):
        if fraction < 1:
            orig_num = len(self.items)
            final_num = int(orig_num * fraction)
            random.shuffle(self.items)

            self.items = self.items[:final_num]

    @staticmethod
    def _check_args(fraction, as_frames, frame_length):
        if fraction < 0 or fraction > 1:
            raise ValueError("fraction should be within [0, 1]")

        if as_frames and frame_length is None:
            raise ValueError("frame_length cannot be None when as_frames=True")
