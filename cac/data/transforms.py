"""Defines class for augmentations on raw signal"""
import warnings
from typing import List, Any, Tuple, Union
from os.path import join
import random
import sys
import numpy as np
from sklearn.decomposition import PCA as _PCA
import noisereduce
import kornia
import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio
from torchaudio import transforms
from torchaudio.transforms import MelSpectrogram, MFCC, AmplitudeToDB, MelScale
from torchaudio.transforms import Spectrogram as _Spectrogram
from torchaudio.transforms import Vol as _Vol
from torchaudio.transforms import TimeStretch as _TimeStretch
from gammatone import filters
from cac.data.base import BaseDataset
from cac.data.utils import read_dataset_from_config
from cac.utils.typing import TransformDict, DatasetConfigDict
from cac.factory import Factory


class Resize:
    """Resize the given input to a particular size

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (128, 40)
    >>> t_signal = Resize((128, 20))(signal)
    >>> t_signal.shape
    (128, 20)

    :param size: desired size after resizing
    :type size: Union[int, Tuple[int]]
    """
    def __init__(self, size: Union[int, Tuple[int]]):
        self.size = size

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        # each input in a batch should atleast be a 3D tensor
        # so, 1D and 2D inputs both need to be converted to 3D

        # if input is 1D, convert it to 2D
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)

        # if the input (or unsqueezed input when signal is 1D)
        # is 2D, convert it to 3D
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(0)

        # needs batch input
        signal = F.interpolate(
            signal.unsqueeze(0), self.size, mode='bilinear',
            align_corners=False)
        return signal.squeeze()


class Log:
    """Return the input in log scale

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (128, 40)
    >>> t_signal = Log(base=2)(signal)
    >>> t_signal.shape
    (128, 40)

    :param base: desired base for the log operation
    :type base: Union[int, str]
    """
    def __init__(self, base: Union[int, str]):
        self._check_params(base)
        self.base = base

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if self.base == 2:
            return signal.log2()
        elif self.base == 10:
            return signal.log10()
        elif self.base == 'natural':
            return signal.log()
        else:
            raise NotImplementedError

    @staticmethod
    def _check_params(base):
        assert base in [2, 10, 'natural'], "only base 2, 10 & e are supported"


class RandomCrop:
    """Randomly crop a fixed-length from an N-dimensional signal

    For input with length < crop_size in the axis specified, the input is
    zero-padded to match the crop_size.

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (257, 200)
    >>> t_signal = RandomCrop(crop_size=100, dim=-1)(signal)
    >>> t_signal.shape
    (257, 100)

    :param crop_size: length to crop from the input signal
    :type crop_size: int
    :param dim: dimension along which to crop, defaults to -1
    :type dim: int, optional
    :param pad_mode: specifies how to handle signal with length less
        than crop_size; `zero` specifies zero padding; defaults to 'zero'
    :type pad_mode: str, optional
    """
    def __init__(
            self, crop_size: int, dim: int = -1,
            pad_mode: str = 'zero'):
        self.crop_size = crop_size
        self.dim = dim
        self.pad_mode = pad_mode

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        ndim = len(signal.shape)
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        n_signal = signal.shape[dim]
        size_diff = abs(n_signal - self.crop_size)

        # handle the case when the input signal is
        # less than the crop duration
        if n_signal < self.crop_size:
            if self.pad_mode == 'zero':
                # randomly pick the length of the zero padding
                # to be applied at the start of the background
                start_pad_length = random.randint(0, size_diff)
                end_pad_length = size_diff - start_pad_length

                # read documentation of torch.nn.functional.pad to understand
                # why this has been done
                padding = []
                for index in reversed(range(ndim)):
                    if index == dim:
                        padding.extend([start_pad_length, end_pad_length])
                        break

                    padding.extend([0, 0])

                signal = F.pad(signal, padding)

            return signal

        start_index = random.randint(0, size_diff)

        # crop a smaller chunk
        signal = signal.transpose(0, dim)
        signal = signal[start_index: start_index + self.crop_size]
        signal = signal.transpose(0, dim)

        return signal


class Sometimes:
    """Applies the specified transform randomly with the specified probability

    Example:
    >>> signal = torch.tensor([0, 1, 2])
    >>> transform = Sometimes(transform_cfg={'name': 'Flip', 'params': {}})
    >>> t_signal = transform(signal)
    >>> t_signal
    tensor([0, 1, 2])
    >>> t_signal = transform(signal)
    >>> t_signal
    tensor([2, 1, 0])

    :param transform_cfg: Config for the transform
    :type transform_cfg: TransformDict
    :param prob: probability of applying the transform; defaults to 0.3
    :type prob: float, optional
    """
    def __init__(self, transform_cfg: TransformDict, prob: float = 0.3):
        self._check_input(transform_cfg, prob)
        self.transform = transform_factory.create(
            transform_cfg['name'], **transform_cfg['params'])
        self.prob = prob

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            signal = self.transform(signal)

        return signal

    @staticmethod
    def _check_input(transform_cfg, prob):
        assert isinstance(transform_cfg, dict)
        assert isinstance(prob, float)


class Flip:
    """Flips the input signal along the specified axis

    Example:
    >>> signal = torch.tensor([0, 1, 2])
    >>> t_signal = Flip()(signal)
    >>> t_signal
    tensor([2, 1, 0])

    :param dim: dimension (s) along which to flip; defaults to 0
    :type dim: Union[int, Tuple], optional
    """
    def __init__(self, dim: Union[int, Tuple] = 0):
        self.dim = self._check_input(dim)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.flip(signal, self.dim)

    @staticmethod
    def _check_input(dim):
        assert isinstance(dim, (int, tuple))

        if not isinstance(dim, tuple):
            return (dim,)

        return dim


class Standardize:
    """Standardizes the input signal

    When `mode=mean-std`, the mean of the signal is subtracted and the
    signal is divided by the standard deviation (along the specified axis).
    If axis=None, then mean/standard deviation is computed for the entire input
    When `mode=min-max`, the input signal `x` is standarized between 0 and 1
    using the following:

    $$ x_t = (x - min(x)) / (max(x) - min(x)) $$

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (257, 100)
    >>> t_signal = Standardize(mode='min-max')(signal)
    >>> t_signal.shape
    (257, 100)

    :param mode: mode of standardization; choices = ['mean-std', 'min-max']
    :type mode: str
    :param use_mean: whether to subtract mean; valid only for `choice=mean-std`
    :type use_mean: bool, defaults to True
    :param use_std: whether to divide by std; valid only for `choice=mean-std`
    :type use_std: bool, defaults to True
    :param mean: mean value to be subtracted from each input; if the value is
        a tensor of length > 1, it is intrepreted as values for each channel.
    :type mean: Union[int, float, torch.Tensor], defaults to None
    :param std: standard deviation value to be divided from each input; if the
        value is a tensor of length > 1, it is intrepreted as values for each
        channel.
    :type std: Union[int, float, torch.Tensor], defaults to None
    :param mean_axis: axis along which to take the mean. If axis=None, the
        value is computed for the entire input; valid only for mode='mean-std';
        ignored if `mean` is not None
    :type mean_axis: int, defaults to None
    :param std_axis: axis along which to compute standard deviation. If
        axis=None, value is computed for the entire input; valid only for
        mode='mean-std'; ignored if `std` is not None
    :type std_axis: int, defaults to None
    """
    def __init__(
            self, mode, use_mean: bool = True, use_std: bool = True,
            mean: Union[int, float, List] = None,
            std: Union[int, float, List] = None,
            mean_axis: int = None, std_axis: int = None):
        self._check_params(mode, mean, std, mean_axis, std_axis)
        self.mode = mode
        self.use_mean = use_mean
        self.use_std = use_std
        self.mean = mean
        self.std = std
        self.mean_axis = mean_axis
        self.std_axis = std_axis

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        self._check_input(signal)
        epsilon = np.finfo(float).eps
        if self.mode == 'mean-std':
            if self.use_mean:
                if self.mean is not None:
                    signal = signal - self.mean
                else:
                    mean_value = self._op_along_axis(
                        signal.mean, self.mean_axis)

                    if self.mean_axis is not None:
                        signal = signal - mean_value
                    else:
                        signal = signal - mean_value

            if self.use_std:
                if self.std is not None:
                    signal = (signal + epsilon) / (self.std + epsilon)
                else:
                    std_value = self._op_along_axis(
                        signal.std, self.std_axis)

                    if self.std_axis is not None:
                        signal = (signal + epsilon) / (std_value + epsilon)
                    else:
                        signal = (signal + epsilon) / (std_value + epsilon)

        elif self.mode == 'min-max':
            signal = (signal - signal.min() + epsilon) / (
                signal.max() - signal.min() + epsilon)

        return signal

    @staticmethod
    def _op_along_axis(op, axis):
        if axis is None:
            return op()
        return op(axis).unsqueeze(axis)

    @staticmethod
    def _check_params(mode, mean, std, mean_axis, std_axis):
        if mode not in ['mean-std', 'min-max']:
            raise ValueError(
                'Valid choices for mode are mean-std and min-max')

        assert isinstance(mean, (int, float, torch.Tensor)) or mean is None
        assert isinstance(std, (int, float, torch.Tensor)) or std is None
        assert isinstance(mean_axis, int) or mean_axis is None
        assert isinstance(std_axis, int) or std_axis is None

    def _check_input(self, signal):
        assert isinstance(signal, torch.Tensor)
        channels = signal.shape[0]
        ndim = len(signal.shape)

        if self.mode == 'mean-std':
            if self.mean is not None:
                mean_channels = 1 if not hasattr(self.mean, 'size') else len(
                    self.mean.view(-1))
                if mean_channels != 1:
                    assert channels == mean_channels
            else:
                if self.mean_axis is not None:
                    _axis = self.mean_axis if self.mean_axis >= 0 else ndim + \
                        self.mean_axis
                    assert _axis >= 0

            if self.std is not None:
                std_channels = 1 if not hasattr(self.std, 'size') else len(
                    self.std.view(-1))
                if std_channels != 1:
                    assert channels == std_channels
            else:
                if self.std_axis is not None:
                    _axis = self.std_axis if self.std_axis >= 0 else ndim + \
                        self.std_axis
                    assert _axis >= 0


class RandomPad:
    """Randomly pad the given input to a fixed-length

    Takes 1D, 2D and 3D tensors as inputs.
    2D and 3D tensors are assumed to be a spectrogram-like signal or the
    STFT.

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (44000,)
    >>> t_signal = RandomPad(target_size=44100)(signal)
    >>> t_signal.shape
    (44100,)

    :param target_size: target output size to obtain after padding
    :type target_size: int
    :param axis: axis along with to apply padding, defaults to -1
    :type axis: str, optional
    :param pad_mode: specifies how to do the padding; `zero` specifies zero
        padding the input to match the length, defaults to 'zero'
    :type pad_mode: str, optional
    """
    def __init__(self, target_size: int, axis: int = -1,
                 pad_mode: str = 'zero'):
        self.target_size = target_size
        self.pad_mode = pad_mode
        self.axis = axis

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        # validate input
        self._check_input(signal)

        # if signal is already of the desired size, simply
        # return the signal
        n_signal = signal.shape[self.axis]

        if n_signal == self.target_size:
            return signal

        ndim = len(signal.shape)
        size_diff = self.target_size - n_signal

        axis = self.axis
        if axis == -1:
            axis = ndim - 1

        if self.pad_mode == 'zero':
            # randomly pick the length of the zero padding
            # to be applied at the start of the background
            start_pad_length = random.randint(0, size_diff)
            end_pad_length = size_diff - start_pad_length

            # read documentation of torch.nn.functional.pad to understand
            # why this has been done
            padding = []
            for index in reversed(range(ndim)):
                if index == axis:
                    padding.extend([start_pad_length, end_pad_length])
                    break

                padding.extend([0, 0])

            signal = F.pad(signal, padding)

        return signal

    def _check_input(self, signal):
        assert isinstance(signal, torch.Tensor)
        ndim = len(signal.shape)
        if ndim > 3:
            raise ValueError(
                "Expected input should either be a 1D/2D/3D tensor")

        if self.axis != -1 and ndim < self.axis + 1:
            raise ValueError("axis={} but input has only {} dimensions".format(
                self.axis, ndim))

        if signal.shape[self.axis] > self.target_size:
            raise ValueError(
                "signal is longer than the target size at axis {}".format(
                    self.axis))


class FixedPad:
    """Pad the end of the given input to match a fixed-length

    Takes 1D, 2D and 3D tensors as inputs.
    2D and 3D tensors are assumed to be a spectrogram-like signal or the
    STFT.

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (44000,)
    >>> t_signal = RandomPad(target_size=44100)(signal)
    >>> t_signal.shape
    (44100,)

    :param target_size: target output size to obtain after padding
    :type target_size: int
    :param axis: axis along with to apply padding, defaults to -1
    :type axis: str, optional
    :param pad_mode: specifies how to do the padding; one of
        ['constant', 'reflect', 'replicate', 'repeat']. `repeat` and `reflect`
        would keep repeating/reflecting the input to match the desired target
        size. Refer to torch.nn.functional.pad for the other values.
    :type pad_mode: str, optional
    """
    def __init__(self, target_size: int, axis: int = -1,
                 pad_mode: str = 'constant'):
        self._check_args(pad_mode)
        self.target_size = target_size
        self.pad_mode = pad_mode
        self.axis = axis

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        # validate input
        self._check_input(signal)

        # if signal is already of the desired size, simply
        # return the signal
        n_signal = signal.shape[self.axis]

        if n_signal == self.target_size:
            return signal

        ndim = len(signal.shape)
        size_diff = self.target_size - n_signal

        axis = self.axis
        if axis == -1:
            axis = ndim - 1

        # correct the input shape
        if self.pad_mode not in ['repeat', 'reflect']:
            if ndim == 1:
                signal = signal.unsqueeze(0)

            if len(signal.shape) == 2:
                signal = signal.unsqueeze(0)

            # read documentation of torch.nn.functional.pad to
            # understand why this has been done
            padding = []
            for index in reversed(range(ndim)):
                if index == axis:
                    # always pad to the end
                    padding.extend([0, size_diff])
                    break

                padding.extend([0, 0])

            # do the padding
            signal = F.pad(signal, padding, self.pad_mode)

            # correct the output shape
            if ndim < 3:
                signal = signal.squeeze(0)

            if ndim < 2:
                signal = signal.squeeze(0)
        else:
            # get the number of times the full signal has to
            # be repeated
            num_full_repeat = 1 + size_diff // n_signal

            # get the partial repeat value
            num_partial_repeat = size_diff % n_signal

            if self.pad_mode == 'repeat':
                partial_signal = signal.transpose(
                    axis, 0)[:num_partial_repeat].transpose(axis, 0)
                chunks = [signal] * num_full_repeat + [partial_signal]
            else:
                flip_signal = torch.flip(signal, (axis,))

                # choose partial chunk either from the start or from the
                # end depending on the input size
                if num_full_repeat % 2:
                    signal_to_reflect = flip_signal
                else:
                    signal_to_reflect = signal

                # offset of 1 is added as we do not want to include the
                # boundary value
                partial_signal = signal_to_reflect.transpose(
                    axis, 0)[1:num_partial_repeat+1].transpose(axis, 0)

                chunks = []

                # accumulate full consecutive reflections
                for index in range(num_full_repeat):
                    if not index % 2:
                        chunks.append(signal)
                    else:
                        chunks.append(flip_signal)

                # add the partial chunk to the end
                chunks.append(partial_signal)

            signal = torch.cat(chunks, dim=axis)

        return signal

    @staticmethod
    def _check_args(pad_mode):
        assert pad_mode in ['constant', 'replicate', 'reflect', 'repeat']

    def _check_input(self, signal):
        assert isinstance(signal, torch.Tensor)
        ndim = len(signal.shape)
        if ndim > 3:
            raise ValueError(
                "Expected input should either be a 1D/2D/3D tensor")

        if self.axis != -1 and ndim < self.axis + 1:
            raise ValueError("axis={} but input has only {} dimensions".format(
                self.axis, ndim))

        if signal.shape[self.axis] > self.target_size:
            raise ValueError(
                "signal is longer than the target size at axis {}".format(
                    self.axis))


class NoiseReduction:
    """Reduces background noise in an input signal based on prototypical noise

    # TODO: Use better ThresholdingBased method for noise clip input
    # TODO: Use noise clip from available noise database(s)

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (10000,)
    >>> noise_clip_method = {
            'name': 'Fixed',
            'params': {
                'start': 0.0,
                'end': 0.5,
                'rate': 16000
            }
        }
    >>> t_signal = NoiseReduction(noise_clip_method)(signal)
    >>> t_signal.shape
    (10000,)

    :param noise_clip_method: dict defining parameters to choose the
        prototypical noise clip
    :type noise_clip_method: TransformDict
    """
    def __init__(self, noise_clip_method: TransformDict):
        super(NoiseReduction, self).__init__()
        self.noise_clip_method = noise_clip_method
        self._check_params(noise_clip_method)

    def __call__(
            self, signal: torch.Tensor,
            as_tensor: bool = True) -> torch.Tensor:
        self._check_input(signal)

        noise_clip = self.get_noise_clip(signal)

        reduced_noise_signal = noisereduce.reduce_noise(
            audio_clip=signal.numpy(),
            noise_clip=noise_clip.numpy())

        if as_tensor:
            reduced_noise_signal = torch.Tensor(reduced_noise_signal)

        return reduced_noise_signal

    def get_noise_clip(self, signal: torch.Tensor) -> torch.Tensor:
        """Returns noise clip based on start and end parameters.

        :param signal: input audio signal
        :type signal: torch.Tensor
        """
        # if method='Fixed', simply pass signal[start:end] as noise clip
        if self.noise_clip_method['name'] == 'Fixed':
            start = self.noise_clip_method['params']['start']
            end = self.noise_clip_method['params']['end']
            rate = self.noise_clip_method['params']['rate']

            start_index = int(start * rate)
            end_index = int(end * rate)

            if start_index < 0 or start_index >= end_index:
                start_index = 0
            if end_index > len(signal) or end_index <= start_index:
                end_index = -1

            return signal[start_index:end_index]

    @staticmethod
    def _check_params(noise_clip_method):
        # For now, only method='Fixed' is implemented
        assert noise_clip_method['name'] == 'Fixed'
        assert 'start' in noise_clip_method['params']
        assert 'end' in noise_clip_method['params']

    @staticmethod
    def _check_input(_input):
        if len(_input.shape) != 1:
            raise ValueError("Expected input is a 1D tensor")


class BackgroundNoise:
    """Adds background noise from a specified dataset to a raw waveform

    Given an input signal `x`, the output `y` is calculated as:

    $y = \alpha * x + (1 - \alpha) * b$

    where `b` is a background signal sampled from the background noise
    dataset and $\alpha$ is sampled uniformly from `[0, max_noise_scale]`.

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (10000)
    >>> dataset_config = {
        'name': 'esc-50', 'version': 'default', 'mode': 'all'}
    >>> t_signal = BackgroundNoise([dataset_config])(signal)
    >>> t_signal.shape
    (10000)

    :param dataset_config: config for the dataset to use as background noise
    :type dataset_config: List[DatasetConfigDict]
    :param min_noise_scale: minimum fraction of background noise in the output
        signal
    :param min_noise_scale: float, defaults to 0.0
    :param max_noise_scale: maximum fraction of background noise in the output
        signal
    :param max_noise_scale: float, defaults to 1.0
    """
    def __init__(
            self, dataset_config: List[DatasetConfigDict],
            min_noise_scale: float = 0.0,
            max_noise_scale: float = 0.3):
        self._check_params(min_noise_scale, max_noise_scale)

        dataset = BaseDataset(dataset_config)
        self.bg_audios = dataset.items
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale

    def __call__(
            self, signal: torch.Tensor,
            return_params: bool = False) -> torch.Tensor:
        # sample a background audio
        bg_audio = random.choice(self.bg_audios)
        bg_audio = bg_audio.load(as_tensor=True)['signal']
        bg_audio = self._check_size(signal, bg_audio)

        # randomly pick the noise fraction
        noise_scale = random.uniform(
            self.min_noise_scale, self.max_noise_scale)
        signal = noise_scale * bg_audio + signal

        if return_params:
            return signal, bg_audio, noise_scale

        return signal

    @staticmethod
    def _check_size(signal, bg_audio):
        signal_size = len(signal)
        bg_size = len(bg_audio)

        # handle the cases when the input signal
        # and background have different lengths
        if signal_size < bg_size:
            start_index = random.randint(0, bg_size - signal_size)

            # sample a smaller chunk from the background
            # matching the length of the input signal
            bg_audio = bg_audio[start_index: start_index + signal_size]

        elif signal_size > bg_size:
            size_diff = signal_size - bg_size

            # randomly pick the length of the zero padding
            # to be applied at the start of the background
            start_pad_length = random.randint(0, size_diff)
            end_pad_length = size_diff - start_pad_length
            padding = (start_pad_length, end_pad_length)
            bg_audio = F.pad(bg_audio, padding)

        return bg_audio

    @staticmethod
    def _check_params(min_noise_scale, max_noise_scale):
        if min_noise_scale < 0 or min_noise_scale > 1:
            raise ValueError("min_noise_scale should be between 0 and 1")

        if max_noise_scale < 0 or max_noise_scale > 1:
            raise ValueError("max_noise_scale should be between 0 and 1")

        if min_noise_scale > max_noise_scale:
            raise ValueError("max_noise_scale should be >= min_noise_scale")

    @staticmethod
    def _check_input(signal):
        if len(signal.shape) != 1:
            raise ValueError("Expected input is a 1D tensor")


class STFT:
    """Computes the Short-Time Fourier Transform (STFT)

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (10000)
    >>> t_signal = STFT(n_fft=400)(signal)
    >>> t_signal.shape
    (201, num_frames)

    :param n_fft: size of Fourier transform
    :type n_fft: int
    :param window: type of window to be used, defaults to None
    :type window: str
    :param kwargs: dictionary containing values for arguments to torch.stft,
        defaults to None
    :type kwargs: dict
    """
    def __init__(self, n_fft: int, window: str = None, **kwargs):
        kwargs['n_fft'] = n_fft

        if window == 'hann':
            kwargs['win_length'] = kwargs.get('win_length', kwargs['n_fft'])
            kwargs['window'] = torch.hann_window(kwargs['win_length'])

        self.kwargs = kwargs

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        # convert to channels-first format as complex
        # coefficients are returned in separate channels
        return torch.stft(signal, **self.kwargs).permute(2, 0, 1)


class ISTFT:
    """Computes the inverse Short-Time Fourier Transform (STFT)

    Input to this should be the output of STFT. Whether the input is
    channels-first or channels-last should be specified in `channels_first`.

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (257, 100)
    >>> t_signal = ISTFT(n_fft=512)(signal)
    >>> t_signal.shape
    (10000,)

    :param n_fft: size of Fourier transform
    :type n_fft: int
    :param window: type of window to be used, defaults to None
    :type window: str
    :param channels_first: whether the input has channels in the first axis,
        defaults to True
    :type channels_first: bool
    :param kwargs: dictionary containing values for arguments to torch.stft,
        defaults to None
    :type kwargs: dict
    """
    def __init__(
            self, n_fft: int, window: str = None,
            channels_first: bool = True, **kwargs):
        kwargs['n_fft'] = n_fft

        if window == 'hann':
            kwargs['win_length'] = kwargs.get('win_length', kwargs['n_fft'])
            kwargs['window'] = torch.hann_window(kwargs['win_length'])

        self.kwargs = kwargs
        self.channels_first = channels_first

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        # input must be in channels-last format
        if self.channels_first:
            signal = signal.permute(1, 2, 0)

        return torch.istft(signal, **self.kwargs)


class Spectrogram:
    """Computes the Spectrogram given an audio signal

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (10000)
    >>> t_signal = Spectrogram(n_fft=400)(signal)
    >>> t_signal.shape
    (201, num_frames)

    :param n_fft: size of Fourier transform
    :type n_fft: int
    :param window: type of window to be used, defaults to None
    :type window: str
    :param power: exponent for the magnitude spectrogram, (must be > 0)
        e.g., 1 for energy, 2 for power, etc. If None, then the complex
        spectrum is returned instead, defaults to 2
    :type power: float or None
    :param kwargs: dictionary containing values for arguments to
        F_audio.spectrogram, defaults to None
    :type kwargs: dict
    """
    def __init__(self, n_fft: int, window: str = None, power: float = 2,
                 **kwargs):
        kwargs['n_fft'] = n_fft
        kwargs['power'] = power
        self.power = power

        if window == 'hann':
            kwargs['win_length'] = kwargs.get('win_length', kwargs['n_fft'])
            kwargs['window_fn'] = torch.hann_window

        self.transform = _Spectrogram(**kwargs)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        signal = self.transform(signal)

        # convert to channels-first format if power is None
        # when the complex coefficients are returned in separate
        # channels
        if self.power is None:
            signal = signal.permute(2, 0, 1)

        return signal


class TimeStretch:
    """Apply time stretch augmentation randomly

    The gain value would be randomly sampled between
    `(min_gain, max_gain)`.

    :param max_rate: maximum rate to be applied
    :type max_rate: float
    :param min_rate: minimum rate to be applied, defaults to 1
    :type min_rate: float, optional
    :param channels_first: whether expected input has channels in the
        first dimension, defaults to True
    :type channels_first: bool, optional
    :param kwargs: arguments for torchaudio.transforms.TimeStretch
    :type kwargs: dict
    """
    def __init__(
            self, max_rate: float, min_rate: float = 1.,
            channels_first: bool = True, **kwargs):
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.channels_first = channels_first
        self.kwargs = kwargs

    def __call__(
            self, signal: torch.Tensor, return_rate=False) -> torch.Tensor:
        rate = random.uniform(self.min_rate, self.max_rate)

        if self.channels_first:
            signal = signal.permute(1, 2, 0)

        signal = _TimeStretch(fixed_rate=rate, **self.kwargs)(signal)

        if self.channels_first:
            signal = signal.permute(2, 0, 1)

        if not return_rate:
            return signal

        return signal, rate


class Flatten:
    """Flatten the input signal to one vector"""
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal.view(-1)


class Squeeze:
    """Removes a dimension (with shape 1) from the input at specified position

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (10, 1)
    >>> t_signal = Squeeze(dim=-1)(signal)
    >>> t_signal.shape
    (10)

    :param dim: axis to be removed
    :type dim: int, defaults to 0
    """
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal.squeeze(self.dim)


class Unsqueeze:
    """Adds a new dimension to the input signal at the specified position

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (10,)
    >>> t_signal = Unsqueeze(dim=-1)(signal)
    >>> t_signal.shape
    (10, 1)

    :param dim: position where the new dimension is to be created
    :type dim: int, defaults to 0
    """
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal.unsqueeze(self.dim)


class Reshape:
    """Reshape the input signal to the specified shape

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (100,)
    >>> t_signal = Reshape(shape=(-1, 10))(signal)
    >>> t_signal.shape
    (10, 10)

    :param shape: desired shape
    :type shape: Union[int, Tuple[int]]
    """
    def __init__(self, shape: Union[int, Tuple[int]]):
        self.shape = shape

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal.reshape(self.shape)


class Volume:
    """Apply volume augmentation

    The gain value would be randomly sampled between
    `(min_gain, max_gain)`.

    :param max_gain: maximum gain to be applied
    :type max_gain: int
    :param min_gain: minimum gain to be applied
    :type min_gain: int
    :param gain_type: type of gain. One of: `amplitude`, `power`, `db`;
        defaults to `amplitude`
    :type gain_type: str, optional
    """
    def __init__(self, max_gain: float, min_gain: float = 0.,
                 gain_type: str = 'amplitude'):
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.gain_type = gain_type

    def __call__(self, signal: torch.Tensor, rgain=False) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        signal = _Vol(gain, self.gain_type)(signal)

        if not rgain:
            return signal

        return signal, gain


class _AxisMasking:
    """Applies masking along a given axis with option for overlapping masks

    :param max_len: maximum length for a single mask; mask length would be
        sampled from [0, max_len]
    :type max_len: int
    :param min_num: minimum number of masks to apply; defaults to 0
    :type min_num: int
    :param max_num: maximum number of masks to apply; number of masks would be
        sampled from [min_num, max_num]; can lead to overlapping masks,
        defaults to 1
    :type max_num: int
    :param mask_value: value to be used as mask, defaults to 0
    :type mask_value: float
    :param deterministic: whether the mask is deterministic; if True,
        `start_index` is mandatory, `num_masks` is not used and `max_len` is
        considered the mask length, defaults to False
    :type deterministic: bool
    :param start_index: index at which to start masking; only used for
        `deterministic=True`; defaults to 0
    :type start_index: int
    """
    def __init__(
            self, max_len: int, min_num: int = 0,
            max_num: int = 1, mask_value: float = 0,
            deterministic: bool = False,
            start_index: int = 0):
        self.max_len = max_len
        self.min_num = min_num
        self.max_num = max_num
        self.mask_value = mask_value
        self.deterministic = deterministic
        self.start_index = start_index

    def __call__(
            self, signal: torch.Tensor, axis: int,
            return_mask_params: bool = False) -> torch.Tensor:
        if not self.deterministic:
            # choose number of masks randomly
            num_masks = random.randint(self.min_num, self.max_num)
        else:
            num_masks = 1

        mask_params = []
        size = signal.shape[axis]

        if size <= self.max_len:
            return signal

        for _ in range(num_masks):
            if not self.deterministic:
                # choose the length of the mask randomly
                mask_len = random.randint(1, self.max_len)

                # choose start index for the mask randomly
                start_index = random.randint(0, size - mask_len)
            else:
                mask_len = self.max_len
                start_index = self.start_index

            # apply the mask
            if axis == 0:
                signal[start_index: start_index + mask_len] = self.mask_value
            elif axis == 1:
                signal[:, start_index: start_index + mask_len] = \
                    self.mask_value
            elif axis == 2:
                signal[:, :, start_index: start_index + mask_len] = \
                    self.mask_value

            mask_params.append([start_index, mask_len])

        if return_mask_params:
            return signal, mask_params

        return signal


class TimeMasking(_AxisMasking):
    """Applies time masking as introduced by Google SpecAugment

    Example:
    >>> signal = compute_spectrogram()
    >>> signal.shape
    (num_freq, num_frames)
    >>> t_signal = TimeMasking(max_len=3)(signal)
    >>> t_signal.shape
    (num_freq, num_frames)

    :param max_len: maximum length for a single mask; mask length would be
        sampled from [0, max_len]
    :type max_len: int
    :param max_num: maximum number of masks to apply; number of masks would be
        sampled from [0, max_num]; can lead to overlapping masks, defaults to 1
    :type max_num: int
    :param mask_value: value to be used as mask, defaults to 0
    :type mask_value: float
    :param deterministic: whether the mask is deterministic; if True,
        `start_index` is mandatory,  `num_masks` is not used and `max_len` is
        considered the mask length, defaults to False
    :type deterministic: bool
    :param start_index: index at which to start masking; only used for
        `deterministic=True`; defaults to 0
    :type start_index: int
    """
    def __init__(
            self, max_len: int, min_num: int = 0,
            max_num: int = 1, mask_value: float = 0,
            deterministic: bool = False,
            start_index: int = 0):
        super(TimeMasking, self).__init__(
            max_len, min_num, max_num, mask_value, deterministic, start_index)

    def __call__(
            self, signal: torch.Tensor,
            return_mask_params: bool = False) -> torch.Tensor:
        # choose number of masks randomly
        ndim = len(signal.shape)
        if ndim == 3:
            # (channels, freq, time)
            axis = 2
        else:
            # (freq, time)
            axis = 1

        return super(TimeMasking, self).__call__(
            signal, axis, return_mask_params)


class FrequencyMasking(_AxisMasking):
    """Applies frequency masking as introduced by Google SpecAugment

    Example:
    >>> signal = compute_spectrogram()
    >>> signal.shape
    (num_freq, num_frames)
    >>> t_signal = FrequencyMasking(max_len=3)(signal)
    >>> t_signal.shape
    (num_freq, num_frames)

    :param max_len: maximum length for a single mask; mask length would be
        sampled from [0, max_len]
    :type max_len: int
    :param max_num: maximum number of masks to apply; number of masks would be
        sampled from [0, max_num]; can lead to overlapping masks, defaults to 1
    :type max_num: int
    :param mask_value: value to be used as mask, defaults to 0
    :type mask_value: float
    :param deterministic: whether the mask is deterministic; if True,
        `start_index` is mandatory,  `num_masks` is not used and `max_len` is
        considered the mask length, defaults to False
    :type deterministic: bool
    :param start_index: index at which to start masking; only used for
        `deterministic=True`; defaults to 0
    :type start_index: int
    """
    def __init__(
            self, max_len: int, min_num: int = 0,
            max_num: int = 1, mask_value: float = 0,
            deterministic: bool = False,
            start_index: int = 0):
        super(FrequencyMasking, self).__init__(
            max_len, min_num, max_num, mask_value, deterministic, start_index)

    def __call__(
            self, signal: torch.Tensor,
            return_mask_params: bool = False) -> torch.Tensor:
        # choose number of masks randomly
        ndim = len(signal.shape)
        if ndim == 3:
            # (channels, freq, time)
            axis = 1
        else:
            # (freq, time)
            axis = 0

        return super(FrequencyMasking, self).__call__(
            signal, axis, return_mask_params)


class GTFB:
    """Computes the GammaTone Filter Bank (GTFB) given an audio signal

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (10000)
    >>> t_signal = GTFB(num_filters=20, low_freq=100)(signal)
    >>> t_signal.shape
    (20, 10000)

    :param num_filters: number of filters in the filter bank
    :type num_filters: int
    :param low_freq: lower cutoff frequency
    :type low_freq: int
    :param sample_rate: sample rate of the input signal
    :type sample_rate: int
    """
    def __init__(
            self, num_filters: int, low_freq: int = 100,
            sample_rate: int = 16000):
        self.num_filters = num_filters
        self.low_freq = low_freq
        self.sample_rate = sample_rate

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        # each filter in the filter bank has a centre frequency associated
        # with it - this creates the list of those center frequencies
        center_freqs = filters.centre_freqs(
            self.sample_rate, self.num_filters, self.low_freq)

        # given center frequencies and the sample rate, this creates the
        # filter bank: (num_filters, 10)
        fcoefs = filters.make_erb_filters(self.sample_rate, center_freqs)

        # creates the gtfb representation
        gtfb = filters.erb_filterbank(signal, fcoefs)
        return torch.FloatTensor(gtfb)


class Rescale:
    """Rescale the given input by a particular value

    Example:
    >>> signal = torch.ones(100) * 255
    >>> signal.max()
    (255)
    >>> t_signal = Rescale(255.)(signal)
    >>> t_signal.max()
    (1.)

    :param value: value to scale the input by, defaults to 1/255.
    :type value: int
    """
    def __init__(self, value: int = 255.):
        self.value = value

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal / self.value


class WhiteNoise:
    """Adds white noise to raw signal

    :param noise_scale: extent of the noise to be added
    :type noise_scale: float
    """
    def __init__(self, noise_scale: float = 0.005):
        self.noise_scale = noise_scale

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(signal) * self.noise_scale
        return signal + noise


class Resample:
    """Resample a signal from one frequency to another

    :param orig_freq: sampling rate of input signal, defaults to 41000
    :type orig_freq: int, optional
    :param new_freq: desired sampling rate, defaults to 16000
    :type new_freq: int, optional
    """
    def __init__(self, orig_freq: int = 41000, new_freq: int = 16000):
        self.resample = transforms.Resample(orig_freq, new_freq)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return self.resample(signal.unsqueeze(0)).squeeze(0)


class PCA:
    """Applies PCA on 2D input (frequency x time) and optionally
    computes norm in time dimension

    # TODO: Implement PCA in pure torch

    Example:
    >>> signal = torch.randn((64, 100))
    >>> t_signal = NormedPCA(n_components=10)(signal)
    >>> t_signal.shape
    (64)

    :param n_components: number of PCA components to consider
    :type n_components: int
    """
    def __init__(self, n_components: int, norm: bool = False,
                 norm_order: int = 1):
        self.pca = _PCA(n_components=n_components)
        self.norm = norm
        self.norm_order = norm_order

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        self._check_input(signal)

        X = signal.numpy()
        Z = self.pca.fit_transform(X)
        Z = torch.tensor(Z)

        if self.norm:
            Z = torch.norm(Z, p=self.norm_order, dim=-1)

        return Z

    @staticmethod
    def _check_input(signal):
        assert isinstance(signal, torch.Tensor)
        assert len(signal.shape) == 2


class AxisNorm:
    """Computes l-p norm along a given dimension of input signal

    Example:
    >>> signal = torch.randn((64, 10))
    >>> t_signal = AxisNorm(order=1)(signal)
    >>> t_signal.shape
    (64)

    :param order: order of norm to be applied
    :type order: int, optional
    :param dim: axis on which norm is to be applied
    :type dim: int, optional
    """
    def __init__(self, order: int = 1, dim: int = -1):
        self.order = order
        self.dim = dim

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        self._check_input(signal)

        return torch.norm(signal, p=self.order, dim=self.dim)

    @staticmethod
    def _check_input(signal):
        assert len(signal.shape) >= 1 and isinstance(signal, torch.Tensor)


class AxisMean:
    """Computes mean along a given dimension of input signal

    Example:
    >>> signal = torch.randn((64, 10))
    >>> t_signal = AxisMean()(signal)
    >>> t_signal.shape
    (64)

    :param dim: axis on which mean is to be applied
    :type dim: int, optional
    """
    def __init__(self, dim: int = -1):
        self.dim = dim

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        self._check_input(signal)

        return torch.mean(signal, dim=self.dim)

    @staticmethod
    def _check_input(signal):
        assert len(signal.shape) >= 1 and isinstance(signal, torch.Tensor)


class Identity:
    """Returns the input signal as is"""
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal


class AddValue:
    """Adds a particular value to the input signal

    :param val: value to be added to the input signal
    :type val: float
    """
    def __init__(self, val: float):
        self.val = val

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal + self.val


class Transpose:
    """Interchange two specified axes in the input

    :param dim0: first dimension to swap
    :type dim0: float
    :param dim1: second dimension to swap
    :type dim1: float
    """
    def __init__(self, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal.transpose(self.dim0, self.dim1)


class Ensemble:
    """Applies a set of parallel transforms on input signal and aggregates them

    :param transforms_cfg: list of parallel transforms to apply on the signal
    :type transforms_cfg: List[List[TransformDict]]
    :param combine: method of aggregation of parallel transforms
    :type combine: str, optional, default to `concat`,
        choices=['concat', 'stack']
    :param dim:  dimension along with to perform the `combine` operation;
        valid for ['concat', 'stack']
    :type dim: int, defaults to 0
    """
    def __init__(self, transforms_cfg: List[List[TransformDict]],
                 combine: str = 'concat', dim: int = 0):
        self.combine = combine
        self.dim = dim

        self.transforms = []
        for transform_list in transforms_cfg:
            sub_transforms = []
            for transform in transform_list:
                sub_transforms.append(
                    transform_factory.create(
                        transform['name'], **transform['params']))

            # add the current pipeline of transforms to list of all
            # transform pipelines
            self.transforms.append(Compose(sub_transforms))

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        transformed_signals = []
        for t in self.transforms:
            transformed_signals.append(t(signal))

        if self.combine == 'concat':
            output_signal = torch.cat(transformed_signals, dim=self.dim)
        elif self.combine == 'stack':
            output_signal = torch.stack(transformed_signals, dim=self.dim)
        else:
            raise NotImplementedError

        return output_signal


class KorniaBase:
    """Base class to apply any kornia augmentation

    :param aug: kornia augmentation class to be used
    :type aug: `kornia.augmentation.AugmentationBase`
    :param kwargs: arguments for the given augmentation
    :type kwargs: dict
    """
    def __init__(
            self, flip_cls: kornia.augmentation.AugmentationBase,
            **kwargs):
        self.transform = flip_cls(**kwargs)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        self._check_input(signal)
        ndim = len(signal.shape)

        # B, C, H, W
        signal = self.transform(signal)
        signal = signal.squeeze()

        if ndim == 2:
            # signal is 2-dimensional
            return signal

        # ndim = 3
        if len(signal.shape) == 2:
            # input signal had 1 as its channel dimension
            # which got removed due to squeeze()
            signal = signal.unsqueeze(0)

        return signal

    @staticmethod
    def _check_input(signal):
        assert isinstance(signal, torch.Tensor)
        assert len(signal.shape) in [2, 3]


class RandomVerticalFlip(KorniaBase):
    """Randomly flips the input along the vertical axis

    :param p: probability of the input being flipped; defaults to 0.5
    :type p: float
    """
    def __init__(
            self, p: float = 0.5):
        super(RandomVerticalFlip, self).__init__(
            kornia.augmentation.RandomVerticalFlip, p=p)


class RandomHorizontalFlip(KorniaBase):
    """Randomly flips the input along the horizontal axis

    :param p: probability of the input being flipped; defaults to 0.5
    :type p: float
    """
    def __init__(
            self, p: float = 0.5):
        super(RandomHorizontalFlip, self).__init__(
            kornia.augmentation.RandomHorizontalFlip, p=p)


class RandomErasing(KorniaBase):
    """
    Erases a random selected rectangle for each image in the batch, putting the
    value to zero. The rectangle will have an area equal to the original image
    area multiplied by a value uniformly sampled between the range
    [scale[0], scale[1]) and an aspect ratio sampled between
    [ratio[0], ratio[1])

    :param p: probability that the random erasing operation will be performed.
        defaults to 0.5
    :type p: float
    :param scale: range of proportion of erased area against input image.
        defaults to (0.02, 0.33)
    :type scale: Tuple[float, float]
    :param ratio: range of aspect ratio of erased area.
        defaults to (0.3, 3.3)
    :type ratio: Tuple[float, float]
    """
    def __init__(
            self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
            ratio: Tuple[float, float] = (0.3, 3.3)):
        super(RandomErasing, self).__init__(
            kornia.augmentation.RandomErasing, p=p,
            scale=scale, ratio=ratio)


class Compose:
    """Composes several transforms together to be applied on raw signal

    :param transforms: list of transforms to apply on the signal
    :type transforms: List[Any]
    """
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            signal = t(signal)
        return signal


class DataProcessor:
    """Defines class for on-the-fly transformations on a given input signal

    :param config: list of dictionaries, each specifying a
        transformation to be applied on a given input signal
    :type config: List[TransformDict]
    """
    def __init__(self, config: List[TransformDict]):
        super(DataProcessor, self).__init__()

        transforms = []
        for transform in config:
            transforms.append(
                transform_factory.create(
                    transform['name'], **transform['params']))

        self.transform = Compose(transforms)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return self.transform(signal)


transform_factory = Factory()
transform_factory.register_builder('Resize', Resize)
transform_factory.register_builder('Rescale', Rescale)
transform_factory.register_builder('MFCC', MFCC)
transform_factory.register_builder('MelSpectrogram', MelSpectrogram)
transform_factory.register_builder('MelScale', MelScale)
transform_factory.register_builder('STFT', STFT)
transform_factory.register_builder('Spectrogram', Spectrogram)
transform_factory.register_builder('TimeMasking', TimeMasking)
transform_factory.register_builder('FrequencyMasking', FrequencyMasking)
transform_factory.register_builder('WhiteNoise', WhiteNoise)
transform_factory.register_builder('Resample', Resample)
transform_factory.register_builder('PCA', PCA)
transform_factory.register_builder('AxisNorm', AxisNorm)
transform_factory.register_builder('AxisMean', AxisMean)
transform_factory.register_builder('Ensemble', Ensemble)
transform_factory.register_builder('AmplitudeToDB', AmplitudeToDB)
transform_factory.register_builder('GTFB', GTFB)
transform_factory.register_builder('BackgroundNoise', BackgroundNoise)
transform_factory.register_builder('NoiseReduction', NoiseReduction)
transform_factory.register_builder('RandomPad', RandomPad)
transform_factory.register_builder('RandomCrop', RandomCrop)
transform_factory.register_builder('Transpose', Transpose)
transform_factory.register_builder('AddValue', AddValue)
transform_factory.register_builder('Flatten', Flatten)
transform_factory.register_builder('Squeeze', Squeeze)
transform_factory.register_builder('Unsqueeze', Unsqueeze)
transform_factory.register_builder('Volume', Volume)
transform_factory.register_builder('Reshape', Reshape)
transform_factory.register_builder('ISTFT', ISTFT)
transform_factory.register_builder('Standardize', Standardize)
transform_factory.register_builder('Identity', Identity)
transform_factory.register_builder('Flip', Flip)
transform_factory.register_builder('Sometimes', Sometimes)
transform_factory.register_builder('TimeStretch', TimeStretch)
transform_factory.register_builder('Log', Log)
transform_factory.register_builder('FixedPad', FixedPad)
transform_factory.register_builder('RandomVerticalFlip', RandomVerticalFlip)
transform_factory.register_builder(
    'RandomHorizontalFlip', RandomHorizontalFlip)
transform_factory.register_builder('RandomErasing', RandomErasing)


class ClassificationAnnotationTransform:
    """
    Transforms the input label to the appropriate target value
    for single-label classification.

    :param classes: list of relevant classes for classification
    :type classes: List[str]
    :param auto_increment: increment each class index by 1 and assign 0
        when there is no intersection between `target` and `classes`; if
        False, raises an error when there is no intersection; defaults to True
    :type auto_increment: bool, optional
    """
    def __init__(self, classes: List[str], auto_increment: bool = True):
        self.classes = classes
        self.auto_increment = auto_increment

    def __call__(self, target:  List[str]) -> int:
        # find the intersection between target and self.classes
        intersection = [
            _target for _target in target if _target in self.classes]

        # ensure that not more than one of the relevant classes
        # is present in the target
        if len(intersection) > 1:
            raise ValueError(
                'target contains more than 1 overlapping class with self.classes')

        # if one intersection, then return the corresponding
        # index incremented by 1 as 0 corresponds to the case
        # when none of the relevant classes are present in the
        # target
        if len(intersection) == 1:
            return self.classes.index(intersection[0]) + int(
                self.auto_increment)

        if self.auto_increment:
            # if there is no intersection, return 0
            return 0

        raise ValueError(
            'target contains has no overlapping class with self.classes')


annotation_factory = Factory()
annotation_factory.register_builder(
    "classification", ClassificationAnnotationTransform)
