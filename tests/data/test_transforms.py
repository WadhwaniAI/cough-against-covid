"""Tests cac.data.transforms.DataProcessor"""
import unittest
import math
import numpy as np
import torch
import random
from torchaudio.transforms import TimeStretch
import librosa
from numpy.testing import assert_array_equal, assert_raises, \
    assert_array_almost_equal
from cac.config import Config
from cac.data.utils import read_dataset_from_config
from cac.data.transforms import DataProcessor, STFT, TimeMasking,\
    FrequencyMasking, BackgroundNoise, RandomCrop, RandomPad, Volume,\
    Flatten, Squeeze, Unsqueeze, Ensemble, Reshape, ISTFT, Standardize, \
    Identity, Flip, Sometimes, TimeStretch, AddValue, Transpose, Log, \
    FixedPad


class DataProcessorTestCase(unittest.TestCase):
    """Class to run tests on DataProcessor"""
    @classmethod
    def setUpClass(cls):
        dataset_config = {
            'name': 'flusense',
            'version': 'default',
            'mode': 'val'
        }
        data_info = read_dataset_from_config(dataset_config)
        cls.signal, cls.rate = librosa.load(data_info['file'][0])
        cls.signal = torch.from_numpy(cls.signal)

    def test_time_stretch(self):
        """Checks TimeStretch"""
        dummy = torch.rand((2, 201, 100))
        processor = TimeStretch(max_rate=1.3, hop_length=160, n_freq=201)

        t_signal, rate = processor(dummy, return_rate=True)
        self.assertTrue(rate >= 1 and rate <= 1.3)
        self.assertEqual(t_signal.shape, (2, 201, math.ceil(100 / rate)))

    def test_log_2(self):
        """Checks Log transform with base 2"""
        dummy = torch.rand((2, 201, 100))
        processor = Log(base=2)
        t_signal = processor(dummy)
        target = np.log2(dummy)
        assert_array_almost_equal(target, t_signal, decimal=5)

    def test_log_natural(self):
        """Checks Log transform with base e"""
        dummy = torch.rand((2, 201, 100))
        processor = Log(base='natural')
        t_signal = processor(dummy)
        target = np.log(dummy)
        assert_array_almost_equal(target, t_signal, decimal=5)

    def test_log_10(self):
        """Checks Log transform with base 10"""
        dummy = torch.rand((2, 201, 100))
        processor = Log(base=10)
        t_signal = processor(dummy)
        target = np.log10(dummy)
        assert_array_almost_equal(target, t_signal, decimal=5)

    def test_identity(self):
        """Checks Identity"""
        dummy = torch.ones(100)
        processor = Identity()

        t_signal = processor(dummy)
        assert_array_equal(t_signal, dummy)

    def test_add_value(self):
        """Checks AddValue"""
        dummy = torch.ones(100)
        processor = AddValue(val=0.1)

        t_signal = processor(dummy)
        assert_array_equal(t_signal, 1.1)

    def test_transpose(self):
        """Checks AddValue"""
        dummy = torch.ones((10, 20))
        processor = Transpose(0, 1)

        t_signal = processor(dummy)
        self.assertEqual(t_signal.shape, (20, 10))

    def test_flip_1d(self):
        """Checks Flip with 1D input"""
        dummy = torch.tensor([0, 1, 2])
        processor = Flip()

        t_signal = processor(dummy)
        assert_array_equal(t_signal, [2, 1, 0])

    def test_flip_2d(self):
        """Checks Flip with 2D input"""
        dummy = torch.tensor([[0, 1, 2], [3, 4, 5]])
        processor = Flip(dim=1)

        t_signal = processor(dummy)
        assert_array_equal(t_signal, [[2, 1, 0], [5, 4, 3]])

    def test_sometimes(self):
        """Checks Sometimes with Flip as transform"""
        dummy = torch.tensor([0, 1, 2])
        transform_cfg = {'name': 'Flip', 'params': {}}
        processor = Sometimes(transform_cfg, prob=0.5)

        transformed = 0
        not_transformed = 0

        random.seed(10)
        for _ in range(10):
            t_signal = processor(dummy)
            try:
                assert_array_equal(t_signal, [2, 1, 0])
                transformed += 1
            except AssertionError:
                not_transformed += 1

        self.assertTrue(not_transformed > 0)
        self.assertTrue(transformed > 0)

    def test_no_transform(self):
        """Checks the case with no signal transform applied"""
        config = []
        processor = DataProcessor(config)

        transformed_signal = processor(self.signal)
        assert_array_equal(self.signal, transformed_signal)

    def test_standardize_with_mean_false(self):
        """Tests Standardize with use_mean=False"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        std = dummy.std()
        t_dummy = Standardize('mean-std', use_mean=False)(dummy)
        target = dummy / std
        assert_array_equal(target, t_dummy)

    def test_standardize_with_std_false(self):
        """Tests Standardize with use_std=False"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        mean = dummy.mean()
        t_dummy = Standardize('mean-std', use_std=False)(dummy)
        target = dummy - mean
        assert_array_equal(target, t_dummy)

    def test_standardize_mean_std(self):
        """Tests Standardize with mean and std specified"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        mean = 0.2
        std = 0.1
        t_dummy = Standardize('mean-std', mean=mean, std=std)(dummy)
        target = (dummy - mean) / std
        assert_array_almost_equal(target, t_dummy, decimal=4)

    def test_standardize_mean_no_std(self):
        """Tests Standardize with only mean specified"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        mean = 0.2
        std = dummy.std()
        t_dummy = Standardize('mean-std', mean=mean)(dummy)
        target = (dummy - mean) / std
        assert_array_almost_equal(target, t_dummy, decimal=4)

    def test_standardize_no_mean_no_std(self):
        """Tests Standardize with neither mean nor std specified"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        mean = dummy.mean()
        std = dummy.std()
        t_dummy = Standardize('mean-std')(dummy)
        target = (dummy - mean) / std
        assert_array_equal(target, t_dummy)

    def test_standardize_mean_std_axis_non_negative(self):
        """Tests Standardize with mean & std specified along axis (axis >= 0)"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        mean = dummy.mean(dim=1)
        std = dummy.std(dim=1)
        t_dummy = Standardize('mean-std', mean_axis=1, std_axis=1)(dummy)
        target = (dummy - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        assert_array_almost_equal(target, t_dummy, decimal=5)

    def test_standardize_mean_std_axis_negative(self):
        """Tests Standardize with mean & std specified along axis (axis < 0)"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        mean = dummy.mean(dim=-1)
        std = dummy.std(dim=-1)
        t_dummy = Standardize('mean-std', mean_axis=-1, std_axis=-1)(dummy)
        target = (dummy - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        assert_array_almost_equal(target, t_dummy, decimal=4)

    def test_standardize_min_max(self):
        """Tests Standardize with mode=min-max"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        _min = dummy.min()
        _max = dummy.max()
        t_dummy = Standardize(mode='min-max')(dummy)
        target = (dummy - _min) / (_max - _min)
        assert_array_almost_equal(target, t_dummy, decimal=4)

    def test_reshape(self):
        """Tests Reshape"""
        dummy = torch.ones(100)
        t_dummy = Reshape(shape=(-1, 10))(dummy)
        self.assertEqual(t_dummy.shape, (10, 10))

    def test_flatten_1d(self):
        """Tests Flatten on 1D input"""
        dummy = torch.ones(10)
        t_dummy = Flatten()(dummy)
        assert_array_equal(dummy, t_dummy)

    def test_flatten_2d(self):
        """Tests Flatten on 2D input"""
        dummy = torch.ones((2, 10))
        t_dummy = Flatten()(dummy)
        self.assertTrue(t_dummy.shape[0], 20)
        self.assertTrue(len(t_dummy.shape), 1)

    def test_squeeze(self):
        """Tests Squeeze"""
        dummy = torch.ones((10, 1))
        t_dummy = Squeeze(dim=-1)(dummy)
        self.assertTrue(t_dummy.shape, (10,))

    def test_unsqueeze(self):
        """Tests Unsqueeze"""
        dummy = torch.ones(10)
        t_dummy = Unsqueeze(dim=-1)(dummy)
        self.assertTrue(t_dummy.shape, (10, 1))

    def test_pca_transform(self):
        """Checks the case with PCA transform applied"""
        n_components = 10
        config = [
            {
                'name': 'PCA',
                'params': {'n_components': n_components, 'norm': True}
            }
        ]

        processor = DataProcessor(config)

        signal = torch.randn((32, 100))
        pca_signal = processor(signal)

        self.assertEqual(pca_signal.shape, (32,))

    def test_axis_norm_1d(self):
        """Checks the case with axis-norm transform applied"""
        config = [
            {
                'name': 'AxisNorm',
                'params': {'order': 2}
            }
        ]

        processor = DataProcessor(config)

        signal = torch.randn((100))
        signal_norm = processor(signal)

        self.assertEqual(signal_norm, torch.norm(signal, p=2, dim=-1))

    def test_axis_norm_2d(self):
        """Checks the case with axis-norm transform applied"""
        config = [
            {
                'name': 'AxisNorm',
                'params': {'order': 2}
            }
        ]

        processor = DataProcessor(config)

        signal = torch.randn((10, 100))
        signal_norm = processor(signal)

        computed_signal_norm = torch.norm(signal, p=2, dim=-1)
        assert_array_equal(signal_norm, computed_signal_norm)

    def test_mean_norm_1d(self):
        """Checks the case with axis-mean transform applied"""
        config = [
            {
                'name': 'AxisMean',
                'params': {}
            }
        ]

        processor = DataProcessor(config)

        signal = torch.randn((100))
        signal_mean = processor(signal)

        self.assertEqual(signal_mean, torch.mean(signal, dim=-1))

    def test_mean_norm_2d(self):
        """Checks the case with axis-mean transform applied"""
        config = [
            {
                'name': 'AxisMean',
                'params': {}
            }
        ]

        processor = DataProcessor(config)

        signal = torch.randn((10, 100))
        signal_mean = processor(signal)

        computed_signal_mean = torch.mean(signal, dim=-1)
        assert_array_equal(signal_mean, computed_signal_mean)

    def test_ensemble_empty(self):
        config = []
        processor = DataProcessor(config)
        signal = torch.randn((10, 100))
        transformed_signal = processor(signal)

        assert_array_equal(transformed_signal, signal)

    def test_ensemble_concat(self):
        transforms_cfg = [
            [
                {
                    'name': 'AxisNorm',
                    'params': {'order': 1}
                }
            ],
            [
                {
                    'name': 'AxisMean',
                    'params': {}
                }
            ]
        ]
        processor = Ensemble(transforms_cfg=transforms_cfg, combine='concat')
        dummy = torch.ones((100, 10))
        t_dummy = processor(dummy)

        self.assertTrue(t_dummy.shape, (200,))

    def test_ensemble_stack(self):
        transforms_cfg = [
            [
                {
                    'name': 'AxisNorm',
                    'params': {'order': 1}
                }
            ],
            [
                {
                    'name': 'AxisMean',
                    'params': {}
                }
            ]
        ]
        processor = Ensemble(transforms_cfg=transforms_cfg, combine='stack')
        dummy = torch.ones((100, 10))
        t_dummy = processor(dummy)

        self.assertTrue(t_dummy.shape, (2, 100))

    def test_ensemble_mean_and_norm(self):
        config = [
            {
                'name': 'Ensemble',
                'params': {
                    'transforms_cfg': [
                        [
                            {
                                'name': 'AxisNorm',
                                'params': {'order': 1}
                            }
                        ],
                        [
                            {
                                'name': 'AxisMean',
                                'params': {}
                            }
                        ]
                    ],
                    'combine': 'concat'
                }
            }
        ]
        processor = DataProcessor(config)
        signal = torch.randn((10, 100))
        transformed_signal = processor(signal)

        subsignals = [torch.norm(signal, p=1, dim=-1), torch.mean(signal, dim=-1)]
        computed_signal = torch.cat(subsignals)

        assert_array_equal(transformed_signal, computed_signal)

    def test_noise_addition_transform(self):
        """Checks the case with noise addition signal transform applied"""
        seed = 0
        noise_scale = 0.005
        config = [
            {
                'name': 'WhiteNoise',
                'params': {'noise_scale': noise_scale}
            }
        ]

        torch.manual_seed(seed)
        processor = DataProcessor(config)
        pred_transformed_signal = processor(self.signal)

        torch.manual_seed(seed)
        noise = torch.randn_like(self.signal) * noise_scale
        gt_transformed_signal = self.signal + noise

        self.assertEqual(self.signal.shape, pred_transformed_signal.shape)
        assert_array_equal(gt_transformed_signal, pred_transformed_signal)

    def test_resize_1d(self):
        """Checks Resize transform with 1D data"""
        size = (1, 1000)
        config = [
            {
                'name': 'Resize',
                'params': {'size': size}
            }
        ]

        processor = DataProcessor(config)
        dummy_signal = torch.zeros(8000)
        transformed_signal = processor(dummy_signal)

        self.assertEqual(transformed_signal.shape, (*size[1:],))

    def test_resize_2d(self):
        """Checks Resize transform with 2D data"""
        size = (128, 20)
        config = [
            {
                'name': 'Resize',
                'params': {'size': size}
            }
        ]

        processor = DataProcessor(config)
        dummy_signal = torch.zeros((128, 50))
        transformed_signal = processor(dummy_signal)

        self.assertEqual(transformed_signal.shape, size)

    def test_resize_3d(self):
        """Checks Resize transform with 3D data"""
        size = (128, 20)
        config = [
            {
                'name': 'Resize',
                'params': {'size': size}
            }
        ]

        processor = DataProcessor(config)
        dummy_signal = torch.zeros((2, 128, 50))
        transformed_signal = processor(dummy_signal)

        self.assertEqual(transformed_signal.shape, (2, *size))

    def test_rescale_transform(self):
        """Checks Resize transform"""
        config = [
            {
                'name': 'Rescale',
                'params': {'value': 255}
            }
        ]

        processor = DataProcessor(config)
        dummy_signal = torch.ones(100) * 255.
        transformed_signal = processor(dummy_signal)

        self.assertTrue(transformed_signal.max(), 1.0)

    def test_spectrogram_transform(self):
        """Tests Spectrogram with no window specified"""
        n_fft = 440
        config = [
            {
                'name': 'Spectrogram',
                'params': {'n_fft': n_fft}
            }
        ]
        # hard coded for this particular file
        num_frames = 21304

        processor = DataProcessor(config)
        signal = processor(self.signal)
        self.assertEqual(signal.shape, (n_fft // 2 + 1, num_frames))

    def test_spectrogram_transform_complex(self):
        """Checks the case with spectrogram transform applied"""
        n_fft = 440
        config = [
            {
                'name': 'Spectrogram',
                'params': {'n_fft': n_fft, 'power': None}
            }
        ]
        # hard coded for this particular file
        num_frames = 21304

        processor = DataProcessor(config)
        signal = processor(self.signal)
        self.assertEqual(signal.shape, (2, n_fft // 2 + 1, num_frames))

    def test_spectrogram_transform_window(self):
        """Tests Spectrogram with 'hann' window specified"""
        n_fft = 440
        config = [
            {
                'name': 'Spectrogram',
                'params': {'n_fft': n_fft, 'window': 'hann'}
            }
        ]
        # hard coded for this particular file
        num_frames = 21304

        processor = DataProcessor(config)
        signal = processor(self.signal)
        self.assertEqual(signal.shape, (n_fft // 2 + 1, num_frames))

    def test_gtfb(self):
        """Tests GTFB"""
        num_filters = 20
        config = [
            {
                'name': 'GTFB',
                'params': {
                    'num_filters': num_filters,
                    'low_freq': 100
                }
            }
        ]

        processor = DataProcessor(config)
        signal = processor(self.signal)
        self.assertEqual(len(signal.shape), 2)
        self.assertEqual(signal.shape[0], num_filters)
        self.assertIsInstance(signal, torch.Tensor)

    def test_melspectrogram_transform(self):
        """Checks the case with mel-spectrogram transform applied"""
        n_mels = 128
        config = [
            {
                'name': 'MelSpectrogram',
                'params': {'n_mels': n_mels, 'win_length': None, 'hop_length': None}
            }
        ]
        # hard coded for this particular file
        num_frames = 23435

        processor = DataProcessor(config)

        signal = processor(self.signal)
        self.assertEqual(signal.shape, (n_mels, num_frames))

    def test_amplitude_to_db_transform(self):
        """Checks the case with amplitude-to-DB (log) transform applied"""
        config = [
            {
                'name': 'AmplitudeToDB',
                'params': {}
            }
        ]
        processor = DataProcessor(config)

        signal = processor(self.signal)
        self.assertEqual(signal.shape, self.signal.shape)

    def test_resample_transform(self):
        """Checks the case with resample transform applied"""
        config = [
            {
                'name': 'Resample',
                'params': {
                    'orig_freq': 32000,
                    'new_freq': 16000
                }
            }
        ]

        processor = DataProcessor(config)
        output = processor(self.signal)

        # number of output samples should be half the number of input
        # samples
        self.assertEqual(output.shape[0], self.signal.shape[0] // 2)

    def test_mfcc_transform(self):
        """Checks the case with MFCC transform applied"""
        config = [
            {
                'name': 'MFCC',
                'params': {'sample_rate': self.rate, 'n_mfcc': 40}
            }
        ]

        # hard coded for this particular file
        n_mfcc = 40
        num_frames = 23435

        processor = DataProcessor(config)
        signal = processor(self.signal)
        self.assertEqual(signal.shape, (n_mfcc, num_frames))

    def test_stft_no_window(self):
        """Tests STFT transform without window specified"""
        processor = STFT(n_fft=400)
        output = processor(self.signal)

        # size should be num_frames x (n_fft // 2 + 1) x 2
        self.assertNotIn('window', processor.kwargs)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 201)

    def test_stft_hann_window(self):
        """Tests STFT transform with `hann` window specified"""
        processor = STFT(n_fft=400, window='hann')
        output = processor(self.signal)

        # size should be num_frames x (n_fft // 2 + 1) x 2
        self.assertIn('window', processor.kwargs)
        self.assertEqual(len(processor.kwargs['window']), 400)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 201)

    def test_istft_no_window(self):
        """Tests ISTFT transform without window specified"""
        dummy = torch.ones((2, 257, 200))
        processor = ISTFT(n_fft=512)
        output = processor(dummy)

        self.assertNotIn('window', processor.kwargs)
        self.assertEqual(len(output.shape), 1)

    def test_istft_hann_window(self):
        """Tests ISTFT transform with `hann` window specified"""
        dummy = torch.ones((2, 257, 200))
        processor = ISTFT(n_fft=512, window='hann')
        output = processor(dummy)

        self.assertIn('window', processor.kwargs)
        self.assertEqual(len(processor.kwargs['window']), 512)
        self.assertEqual(len(output.shape), 1)

    def test_istft_channels_last(self):
        """Tests ISTFT transform with input in channels_last format"""
        dummy = torch.ones((257, 200, 2))
        processor = ISTFT(n_fft=512, channels_first=False)
        output = processor(dummy)

        self.assertEqual(len(output.shape), 1)

    def test_time_masking_small_size(self):
        """Tests TimeMasking with mask size < len(input)"""
        dummy_input = torch.ones((128, 20))
        processor = TimeMasking(max_len=50, min_num=1, max_num=5)
        signal = processor(dummy_input, return_mask_params=True)

        assert_array_equal(signal, dummy_input)

    def test_time_masking_2d(self):
        """Tests TimeMasking with 2D input"""
        dummy_input = torch.ones((128, 20))
        processor = TimeMasking(max_len=5, min_num=1, max_num=5)
        signal, mask_params = processor(dummy_input, return_mask_params=True)
        signal = signal.numpy()

        for mask_param in mask_params:
            start_index, length = mask_param
            assert_array_equal(signal[:, start_index: start_index + length], 0)

    def test_time_masking_3d(self):
        """Tests TimeMasking with 3D input"""
        dummy_input = torch.ones((2, 128, 20))
        processor = TimeMasking(max_len=5, min_num=1, max_num=5)
        signal, mask_params = processor(dummy_input, return_mask_params=True)
        signal = signal.numpy()

        for mask_param in mask_params:
            start_index, length = mask_param
            assert_array_equal(signal[:, :, start_index: start_index + length], 0)

    def test_frequency_masking_2d(self):
        """Tests FrequencyMasking with 2D input"""
        dummy_input = torch.ones((128, 20))
        processor = FrequencyMasking(max_len=5, min_num=1, max_num=5)
        signal, mask_params = processor(dummy_input, return_mask_params=True)
        signal = signal.numpy()

        for mask_param in mask_params:
            start_index, length = mask_param
            assert_array_equal(signal[start_index: start_index + length], 0)

    def test_frequency_masking_3d(self):
        """Tests FrequencyMasking with 3D input"""
        dummy_input = torch.ones((2, 128, 20))
        processor = FrequencyMasking(max_len=5, min_num=1, max_num=5)
        signal, mask_params = processor(dummy_input, return_mask_params=True)
        signal = signal.numpy()

        for mask_param in mask_params:
            start_index, length = mask_param
            assert_array_equal(signal[:, start_index: start_index + length], 0)

    def test_frequency_masking_3d_deterministic(self):
        """Tests FrequencyMasking with 3D input with determistic=True"""
        dummy_input = torch.ones((2, 128, 20))
        processor = FrequencyMasking(max_len=64, start_index=0, deterministic=True)
        signal = processor(dummy_input)
        assert_array_equal(signal[:, :64], 0)

    def test_random_crop_valid_longer(self):
        """Tests RandomCrop with valid input shape and length > crop_size"""
        dummy_input = torch.ones(441000)
        processor = RandomCrop(crop_size=44100)
        signal = processor(dummy_input)

        self.assertEqual(signal.shape[0], 44100)

    def test_random_crop_valid_equal(self):
        """Tests RandomCrop with valid input shape and length = crop_size"""
        dummy_input = torch.ones(44100)
        processor = RandomCrop(crop_size=44100)
        signal = processor(dummy_input)

        self.assertEqual(signal.shape[0], 44100)

    def test_random_crop_valid_shorter(self):
        """Tests RandomCrop with valid input shape and length < crop_size"""
        dummy_input = torch.ones(22050)
        processor = RandomCrop(crop_size=44100)
        signal = processor(dummy_input)

        self.assertEqual(signal.shape[0], 44100)

    def test_random_crop_2d_valid_longer(self):
        """Tests RandomCrop with 2d input and length > crop_size"""
        dummy_input = torch.ones((257, 200))
        processor = RandomCrop(crop_size=100, dim=-1)
        signal = processor(dummy_input)

        self.assertEqual(signal.shape, (257, 100))

    def test_random_crop_2d_valid_equal(self):
        """Tests RandomCrop with 2d input and length = crop_size"""
        dummy_input = torch.ones((257, 100))
        processor = RandomCrop(crop_size=100, dim=-1)
        signal = processor(dummy_input)

        self.assertEqual(signal.shape, (257, 100))

    def test_random_crop_2d_valid_shorter(self):
        """Tests RandomCrop with 2d input and length < crop_size"""
        dummy_input = torch.ones((257, 50))
        processor = RandomCrop(crop_size=100, dim=-1)
        signal = processor(dummy_input)

        self.assertEqual(signal.shape, (257, 100))

    def test_random_pad_valid_1d(self):
        """Tests RandomPad with valid 1D input"""
        dummy_input = torch.ones(44000)
        processor = RandomPad(target_size=44100)
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[0], 44100)

    def test_random_pad_valid_2d(self):
        """Tests RandomPad with valid 2D input"""
        dummy_input = torch.ones((128, 23))
        processor = RandomPad(target_size=30)
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[-1], 30)

    def test_random_pad_valid_3d(self):
        """Tests RandomPad with valid 3D input"""
        dummy_input = torch.ones((2, 128, 23))
        processor = RandomPad(target_size=30)
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[-1], 30)

    def test_random_pad_valid_equal(self):
        """Tests RandomPad with input length = target_size"""
        dummy_input = torch.ones(44100)
        processor = RandomPad(target_size=44100)
        signal = processor(dummy_input)
        assert_array_equal(signal.numpy(), dummy_input.numpy())

    def test_random_pad_invalid_shape(self):
        """Tests RandomPad with invalid input shape"""
        dummy_input = torch.ones((2, 2, 2, 2))
        processor = RandomPad(target_size=44100)
        with self.assertRaises(ValueError):
            signal = processor(dummy_input)

    def test_random_pad_invalid_axis(self):
        """Tests RandomPad with invalid axis"""
        dummy_input = torch.ones(44000)
        processor = RandomPad(target_size=44100, axis=1)
        with self.assertRaises(ValueError):
            signal = processor(dummy_input)

    def test_random_pad_invalid_longer(self):
        """Tests RandomPad with invalid input length"""
        dummy_input = torch.ones((44101))
        processor = RandomPad(target_size=44100)
        with self.assertRaises(ValueError):
            signal = processor(dummy_input)

    def test_fixed_pad_valid_1d_repeat(self):
        """Tests FixedPad with valid 1D input with repeat"""
        dummy_input = torch.rand(44000)
        processor = FixedPad(target_size=44100, pad_mode='repeat')
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[0], 44100)
        assert_array_equal(signal[44000:], signal[:100])

    def test_fixed_pad_valid_1d_repeat_multiple_full(self):
        """Tests FixedPad with valid 1D input with repeat multiple"""
        dummy_input = torch.rand(20000)
        processor = FixedPad(target_size=44100, pad_mode='repeat')
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[0], 44100)
        assert_array_equal(signal[20000: 40000], dummy_input)
        assert_array_equal(signal[40000:], dummy_input[:4100])

    def test_fixed_pad_valid_2d_repeat(self):
        """Tests FixedPad with valid 2D input with repeat"""
        dummy_input = torch.rand((128, 23))
        processor = FixedPad(target_size=30, pad_mode='repeat')
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[-1], 30)
        assert_array_equal(signal[:, 23:], signal[:, :7])

    def test_fixed_pad_valid_3d_repeat(self):
        """Tests FixedPad with valid 3D input with repeat"""
        dummy_input = torch.rand((2, 128, 23))
        processor = FixedPad(target_size=30, pad_mode='repeat')
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[-1], 30)
        assert_array_equal(signal[:, :, 23:], signal[:, :, :7])

    def test_fixed_pad_valid_1d_reflect(self):
        """Tests FixedPad with valid 1D input with reflect"""
        dummy_input = torch.rand(44000)
        processor = FixedPad(target_size=44100, pad_mode='reflect')
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[0], 44100)
        assert_array_equal(
            signal[44000:], torch.flip(dummy_input[-101:-1], (0,)))

    def test_fixed_pad_valid_1d_reflect_multiple_full(self):
        """Tests FixedPad with valid 1D input with reflect multiple"""
        dummy_input = torch.rand(20000)
        processor = FixedPad(target_size=44100, pad_mode='reflect')
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[0], 44100)
        assert_array_equal(signal[20000: 40000], torch.flip(dummy_input, (0,)))
        assert_array_equal(signal[40000:], dummy_input[1:4101])

    def test_fixed_pad_valid_2d_reflect(self):
        """Tests FixedPad with valid 2D input with reflect"""
        dummy_input = torch.rand((128, 23))
        processor = FixedPad(target_size=30, pad_mode='reflect')
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[-1], 30)
        assert_array_equal(
            signal[:, 23:], torch.flip(dummy_input[:, -8:-1], (1,)))

    def test_fixed_pad_valid_3d_reflect(self):
        """Tests FixedPad with valid 3D input with reflect"""
        dummy_input = torch.rand((2, 128, 23))
        processor = FixedPad(target_size=30, pad_mode='reflect')
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[-1], 30)
        assert_array_equal(
            signal[:, :, 23:], torch.flip(dummy_input[:, :, -8:-1], (2,)))

    def test_fixed_pad_valid_equal(self):
        """Tests FixedPad with input length = target_size for reflect"""
        dummy_input = torch.rand(44100)
        processor = FixedPad(target_size=44100, pad_mode='reflect')
        signal = processor(dummy_input)
        assert_array_equal(signal.numpy(), dummy_input.numpy())

    def test_fixed_pad_valid_1d(self):
        """Tests FixedPad with valid 1D input"""
        dummy_input = torch.ones(44000)
        processor = FixedPad(target_size=44100)
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[0], 44100)
        assert_array_equal(signal[44000:], 0)

    def test_fixed_pad_valid_2d(self):
        """Tests FixedPad with valid 2D input"""
        dummy_input = torch.ones((128, 23))
        processor = FixedPad(target_size=30)
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[-1], 30)
        assert_array_equal(signal[:, 23:], 0)

    def test_fixed_pad_valid_3d(self):
        """Tests FixedPad with valid 3D input"""
        dummy_input = torch.ones((2, 128, 23))
        processor = FixedPad(target_size=30)
        signal = processor(dummy_input)
        self.assertEqual(signal.shape[-1], 30)
        assert_array_equal(signal[:, :, 23:], 0)

    def test_fixed_pad_valid_equal(self):
        """Tests FixedPad with input length = target_size"""
        dummy_input = torch.ones(44100)
        processor = FixedPad(target_size=44100)
        signal = processor(dummy_input)
        assert_array_equal(signal.numpy(), dummy_input.numpy())

    def test_fixed_pad_invalid_shape(self):
        """Tests FixedPad with invalid input shape"""
        dummy_input = torch.ones((2, 2, 2, 2))
        processor = FixedPad(target_size=44100)
        with self.assertRaises(ValueError):
            signal = processor(dummy_input)

    def test_fixed_pad_invalid_axis(self):
        """Tests FixedPad with invalid axis"""
        dummy_input = torch.ones(44000)
        processor = FixedPad(target_size=44100, axis=1)
        with self.assertRaises(ValueError):
            signal = processor(dummy_input)

    def test_fixed_pad_invalid_longer(self):
        """Tests FixedPad with invalid input length"""
        dummy_input = torch.ones((44101))
        processor = FixedPad(target_size=44100)
        with self.assertRaises(ValueError):
            signal = processor(dummy_input)

    def test_volume_augmentation(self):
        """Tests Volume augmentation"""
        dummy_input = torch.ones(44100) / 2
        processor = Volume(max_gain=2, gain_type='amplitude')
        signal, gain = processor(dummy_input, rgain=True)
        assert_array_equal(
            signal.numpy(), torch.clamp(dummy_input * gain, -1, 1).numpy())

    def test_background_noise_signal_shorter(self):
        """Tests BackgroundNoise for signal shorter than background"""
        dataset_config = [{
            'name': 'esc-50',
            'version': 'default',
            'mode': 'all'
        }]
        max_noise_scale = 0.3

        processor = BackgroundNoise(
            dataset_config, max_noise_scale=max_noise_scale)
        dummy_signal = torch.ones(100)
        dummy_bg = torch.ones(150)
        t_bg = processor._check_size(dummy_signal, dummy_bg)
        self.assertEqual(len(dummy_signal), len(t_bg))

    def test_background_noise_signal_longer(self):
        """Tests BackgroundNoise for signal longer than background"""
        dataset_config = [{
            'name': 'esc-50',
            'version': 'default',
            'mode': 'all'
        }]
        max_noise_scale = 0.3

        processor = BackgroundNoise(
            dataset_config, max_noise_scale=max_noise_scale)
        dummy_signal = torch.ones(150)
        dummy_bg = torch.ones(100)
        t_bg = processor._check_size(dummy_signal, dummy_bg)
        self.assertEqual(len(dummy_signal), len(t_bg))

    def test_background_noise(self):
        """Tests BackgroundNoise"""
        dataset_config = [{
            'name': 'esc-50',
            'version': 'default',
            'mode': 'all'
        }]
        min_noise_scale = 0.3
        max_noise_scale = 0.6

        processor = BackgroundNoise(
            dataset_config, min_noise_scale=min_noise_scale,
            max_noise_scale=max_noise_scale)
        dummy_signal = torch.ones(150)

        for _ in range(10):
            t_signal, bg_signal, noise_scale = processor(
                dummy_signal, return_params=True)
            self.assertTrue(
                noise_scale >= min_noise_scale and
                noise_scale <= max_noise_scale)
            assert_array_equal(
                t_signal, dummy_signal + noise_scale * bg_signal)

    def test_noise_reduction(self):
        """Tests NoiseReduction"""

        cfg = [
            {
                'name': 'NoiseReduction',
                'params': {
                    'noise_clip_method': {
                        'name': 'Fixed',
                        'params': {'start': 0.0, 'end': 0.5, 'rate': self.rate}
                    }
                }
            }
        ]

        processor = DataProcessor(cfg)
        noise_reduced_signal = processor(self.signal)

        self.assertEqual(len(self.signal), len(noise_reduced_signal))
        self.assertIsInstance(noise_reduced_signal, torch.Tensor)

    def test_random_vertical_flip_2d(self):
        """Tests RandomVerticalFlip on 2D input"""
        dummy = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3
        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3

        config = [
            {
                'name': 'RandomVerticalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

        config = [
            {
                'name': 'RandomVerticalFlip',
                'params': {
                    'p': 0.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(dummy, t_dummy)

    def test_random_vertical_flip_3d_1_channel(self):
        """Tests RandomVerticalFlip on 3D input with 1 channel"""
        dummy = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])     # 3 x 3
        dummy = dummy.unsqueeze(0)               # 1 x 3 x 3
        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3
        expected = expected.unsqueeze(0)         # 1 x 3 x 3

        config = [
            {
                'name': 'RandomVerticalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

    def test_random_vertical_flip_3d_2_channels(self):
        """Tests RandomVerticalFlip on 3D input with 2 channels"""
        dummy = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])           # 3 x 3
        dummy = dummy.unsqueeze(0).expand(2, -1, -1)   # 2 x 3 x 3
        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])             # 3 x 3
        expected = expected.unsqueeze(0).expand(2, -1, -1)  # 2 x 3 x 3

        config = [
            {
                'name': 'RandomVerticalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

    def test_random_horizontal_flip_2d(self):
        """Tests RandomHorizontalFlip on 2D input"""
        dummy = torch.tensor([[1., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.]])  # 3 x 3
        expected = torch.tensor([[0., 0., 1.],
                                 [0., 0., 1.],
                                 [0., 0., 0.]])  # 3 x 3

        config = [
            {
                'name': 'RandomHorizontalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

        config = [
            {
                'name': 'RandomHorizontalFlip',
                'params': {
                    'p': 0.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(dummy, t_dummy)

    def test_random_horizontal_flip_3d_1_channel(self):
        """Tests RandomHorizontalFlip on 3D input with 1 channel"""
        dummy = torch.tensor([[1., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.]])  # 3 x 3
        dummy = dummy.unsqueeze(0)               # 1 x 3 x 3
        expected = torch.tensor([[0., 0., 1.],
                                 [0., 0., 1.],
                                 [0., 0., 0.]])  # 3 x 3
        expected = expected.unsqueeze(0)         # 1 x 3 x 3

        config = [
            {
                'name': 'RandomHorizontalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

    def test_random_horizontal_flip_3d_2_channels(self):
        """Tests RandomHorizontalFlip on 3D input with 2 channels"""
        dummy = torch.tensor([[1., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.]])  # 3 x 3
        dummy = dummy.unsqueeze(0).expand(2, -1, -1)   # 2 x 3 x 3
        expected = torch.tensor([[0., 0., 1.],
                                 [0., 0., 1.],
                                 [0., 0., 0.]])  # 3 x 3
        expected = expected.unsqueeze(0).expand(2, -1, -1)  # 2 x 3 x 3

        config = [
            {
                'name': 'RandomHorizontalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

    def test_random_erasing_2d(self):
        """Tests RandomErasing on 2D input"""
        dummy = torch.rand((3, 3))  # 3 x 3

        config = [
            {
                'name': 'RandomErasing',
                'params': {
                    'p': 1.,
                    'scale': (.001, .001),
                    'ratio': (.2, .2)
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        self.assertEqual(t_dummy.shape, dummy.shape)

        config = [
            {
                'name': 'RandomErasing',
                'params': {
                    'p': 0.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(dummy, t_dummy)


if __name__ == "__main__":
    unittest.main()
