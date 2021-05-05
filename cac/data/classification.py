"""
Defines ClassificationDataset class which is used for classification tasks
where each input has a single output.
"""
from os.path import join
from typing import Tuple, List, Union

import torch
from torch.utils.data import Dataset

from cac.data.audio import AudioItem
from cac.data.base import BaseDataset
from cac.data.transforms import DataProcessor, ClassificationAnnotationTransform
from cac.utils.typing import DatasetConfigDict


class ClassificationDataset(BaseDataset):
    """Dataset class for single-label classification

    If `as_frames=True`, each signal is divided into fixed-length frames. If the
    signal is shorter than the `frame_length`, the signal is simply returned as a
    frame. When signal is longer than `frame_length`, the last frame is most likely
    to not be equal to the `frame_length`. So, the last `frame_length` milliseconds
    make the final frame.

    :param dataset_config: defines the config for the
        data to be loaded. The config is specified by a list of dict, with each dict
        representing: (dataset_name, dataset_version, mode [train, test, val])
    :type dataset_config: DatasetConfigDict
    :param target_transform: defines the transformation
        to be applied on the raw targets to make them processable.
    :type target_transform: ClassificationAnnotationTransform
    :param signal_transform: defines the list of transformations to be applied on the
        raw signals where each transform is defined by a TransformDict object
    :type signal_transform: DataProcessor
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
                 target_transform: ClassificationAnnotationTransform = None,
                 signal_transform: DataProcessor = None,
                 fraction: float = 1.0, as_frames: bool = False,
                 frame_length: int = None, hop_length: int = 500,
                 min_length: int = 100):
        super(ClassificationDataset, self).__init__(
            dataset_config, fraction, as_frames, frame_length,
            hop_length, min_length)
        self.target_transform = target_transform
        self.signal_transform = signal_transform

    def __getitem__(
            self, index: int, as_tensor=True
            ) -> Tuple[torch.Tensor, Union[List[str], int]]:
        item = self.items[index]

        self._check_item(index, item)

        audio = item.load(as_tensor=as_tensor)
        signal = audio['signal']

        if self.signal_transform is not None:
            signal = self.signal_transform(signal)

        label = item.label['classification']
        if self.target_transform is not None:
            label = self.target_transform(label)

        instance = {
            'signal': signal,
            'label': label,
            'item': item
        }

        return instance

    def _check_item(self, index: int, item: AudioItem):
        assert 'classification' in item.label,\
            "Item at index {} does not contain 'classification' in label".format(index)

    def __len__(self):
        return len(self.items)


class ClassificationDatasetBuilder:
    """Builds a ClassificationDataset object"""
    def __call__(self, mode: str, dataset_config: List[dict], **kwargs):
        """Builds a ClassificationDataset object

        :param mode: mode/split to load; one of {'train', 'test', 'val'}
        :type mode: str
        :param dataset_config: list of dictionaries, each containing
            (name, version, mode) corresponding to a dataset
        :type dataset_config: List[dict]
        :param **kwargs: dictionary containing values corresponding to the arguments of
            the ClassificationDataset class
        :type **kwargs: dict
        :returns: a ClassificationDataset object
        """
        for i, config in enumerate(dataset_config):
            dataset_config[i]['mode'] = mode

        kwargs['dataset_config'] = dataset_config
        self._instance = ClassificationDataset(**kwargs)
        return self._instance
