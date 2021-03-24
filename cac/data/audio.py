"""Defines AudioItem class which contains all information related to a given audio file."""

from os.path import join
from typing import Tuple, Union

import librosa
import numpy as np
import torch
from termcolor import colored

from cac.utils.logger import color


class AudioItem(object):
    """Class representing an audio file.

    :param path: Path to the audio file
    :type path: str
    :param label: Annotations for the audio file in the format {'task': task_labels},
        defaults to None
    :type label: dict
    :param start: Start of the segment to consider in the audio file
    :type start: float
    :param end: end of the segment to consider in the audio file
    :type end: float
    """
    def __init__(self, path: str, label: dict = None, start: float = None, end: float = None):
        self.path = path
        self.label = label
        self.start = start
        self.end = end

    def load(
            self, as_tensor: bool = False
            ) -> Tuple[Union[np.ndarray, torch.Tensor], float]:
        """Loads signal and sampling rate of a given audio file

        :param as_tensor: whether to return the signal as torch.Tensor, defaults to False
        :type as_tensor: bool

        :returns: dict containing signal, rate
        """
        signal, rate = librosa.load(self.path, sr=44100)

        # cut the signal from start time to end time
        start_idx = 0 if self.start is None else np.round(self.start * rate).astype(int)
        end_idx = -1 if self.end is None else np.round(self.end * rate).astype(int)
        signal = signal[start_idx: end_idx]

        if as_tensor:
            signal = torch.Tensor(signal)

        audio = {
            'signal': signal,
            'rate': rate
        }

        return audio

    @property
    def sampling_rate(self) -> float:
        """Returns the sampling rate of the audio

        :returns: sampling rate in Hz
        """
        if not hasattr(self, '_sampling_rate'):
            data = self.load()
            self._sampling_rate = data['rate']
        return self._sampling_rate

    def info(self) -> dict:
        """Returns information related to the audio

        :returns: dictionary containing values for different attributes
        """
        value = {
            'path': self.path
        }
        if self.label is not None:
            value.update({'label': self.label})

        return value
