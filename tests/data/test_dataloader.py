"""Tests cac.data.dataloader.py"""
import torch
import numpy as np
import unittest
from cac.config import Config
from cac.data.audio import AudioItem
from cac.data.dataloader import get_dataloader, classification_collate, unsupervised_collate
from cac.data.utils import read_dataset_from_config


class DataloaderTestCase(unittest.TestCase):
    """Class to check the creation of DataLoader"""
    def test_unsupervised_dataloader_2d(self):
        """Test get_dataloader for unsupervised learning task with each input being 2D"""
        n_mfcc = 64
        config = Config('defaults/unsupervised.yml')
        cfg = config.data
        batch_size = 8

        dataloader, _ = get_dataloader(
            cfg, 'test', batch_size=batch_size, shuffle=True, drop_last=True)

        iterator = iter(dataloader)
        batch = next(iterator)
        signals, labels = batch['signals'], batch['labels']

        self.assertIsInstance(signals, torch.Tensor)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(labels[0], dict)
        self.assertTrue('patient_id' in labels[0].keys())
        self.assertTrue('dataset-name' in labels[0].keys())
        self.assertEqual(signals[0].dtype, torch.float32)
        self.assertEqual(len(signals), len(labels))

        # MFCC output shape: (8, 64, num_frames)
        # NormedPCA outputs shape: (8, 64)
        # AxisMean outputs shape: (8, 64)
        # Final output shape (after concatentation): (8, 64 * 2)
        self.assertEqual(len(signals.shape), 2)
        self.assertEqual(signals.shape, (batch_size, 2 * n_mfcc))

    def test_binary_classification_dataloader_1dcnn(self):
        """Test get_dataloader for binary classification task with each input being 1D"""
        resized_length = 1000
        config = Config('defaults/1d.yml')
        config.data['dataset']['params'] = {
            'val': {
                'fraction': 0.1
            }
        }
        cfg = config.data
        batch_size = 8

        dataloader, _ = get_dataloader(
            cfg, 'val', batch_size=batch_size, shuffle=True, drop_last=True)

        iterator = iter(dataloader)
        batch = next(iterator)
        signals, labels = batch['signals'], batch['labels']

        self.assertIsInstance(signals, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(signals.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(len(signals), len(labels))
        self.assertEqual(len(signals.shape), 3)
        self.assertTrue(signals.shape, (batch_size, 1, resized_length))

    def test_binary_classification_dataloader_classification_sampler(self):
        """Test get_dataloader for binary classification task with classification sampler
        TODO: pass this test when datasets ready
        """
        config = Config('defaults/wiai.yml')
        config.data['sampler']['val'] = {
            'name': 'classification',
            'params': {
                'mode': 'balanced'
            }
        }
        cfg = config.data
        batch_size = 8

        dataloader, _ = get_dataloader(
            cfg, 'val', batch_size=batch_size, shuffle=True, drop_last=True)

        iterator = iter(dataloader)
        batch = next(iterator)
        signals, labels = batch['signals'], batch['labels']

        self.assertIsInstance(signals, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(signals.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(len(signals), len(labels))
        self.assertEqual(len(signals.shape), 4)
        self.assertTrue(signals.shape, (batch_size, 1, 257, 100))
        self.assertTrue(len(labels[labels == 0]), 4)
        self.assertTrue(len(labels[labels == 0]), len(labels[labels == 1]))

    def test_binary_classification_dataloader_2d(self):
        """Test get_dataloader for binary classification task with each input being 2D"""
        n_mels = 128
        config = Config('default.yml')
        config.data['dataset']['params']['val']['fraction'] = 0.1
        cfg = config.data
        batch_size = 8

        dataloader, _ = get_dataloader(
            cfg, 'val', batch_size=batch_size, shuffle=True, drop_last=True)

        iterator = iter(dataloader)
        batch = next(iterator)
        signals, labels = batch['signals'], batch['labels']

        self.assertIsInstance(signals, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(signals.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(len(signals), len(labels))
        self.assertEqual(len(signals.shape), 4)
        self.assertTrue(signals.shape, (batch_size, 1, n_mels, 20))

    def test_binary_classification_dataloader_3d(self):
        """Test get_dataloader for binary classification task with each input being 3D
        TODO: pass this test when datasets ready
        """
        n_fft = 512
        config = Config('default-stft.yml')
        config.data['dataset']['params']['val']['fraction'] = 0.1
        cfg = config.data
        batch_size = 8

        dataloader, _ = get_dataloader(
            cfg, 'val', batch_size=batch_size, shuffle=True, drop_last=True)

        iterator = iter(dataloader)
        batch = next(iterator)
        signals, labels = batch['signals'], batch['labels']

        self.assertIsInstance(signals, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(signals.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(len(signals), len(labels))
        self.assertEqual(len(signals.shape), 4)
        self.assertTrue(signals.shape, (batch_size, 2, n_fft // 2 + 1, 20))

    def test_fraction(self):
        """Test get_dataloader for binary classification task"""
        config = Config('default.yml')
        config.data['dataset']['params']['val']['fraction'] = 0.1
        cfg = config.data
        batch_size = 8

        cfg['dataset']['params']['val']['fraction'] = 0.5
        cfg['dataset']['params']['val']['as_frames'] = False
        _, dataset = get_dataloader(
            cfg, 'val', batch_size=batch_size, shuffle=True, drop_last=True)

        data_info = read_dataset_from_config(cfg['dataset']['config'][0])
        self.assertEqual(len(data_info['file']) // 2, len(dataset.items))

    def test_unsupervised_collate_1D_no_zero_pad_no_stack(self):
        """Tests unsupervised_collate with zero_pad=False with stack=False on 1D waveform"""
        num = 4
        paths = [''] * num
        labels = np.ones(num)
        items = [AudioItem(path, label) for path, label in zip(paths, labels)]
        signals = [
            torch.ones((20,)),
            torch.ones((22,)),
            torch.ones((24,)),
            torch.ones((23,))
        ]

        batch = []
        for index in range(num):
            batch.append({
                'signal': signals[index],
                'label': labels[index],
                'item': items[index],
            })

        t_batch = unsupervised_collate(
            batch, zero_pad=False, stack=False)
        self.assertIsInstance(t_batch['signals'], list)
        self.assertEqual(len(t_batch['signals']), num)
        self.assertEqual(len(t_batch['signals'][0]), 20)
        self.assertEqual(len(t_batch['signals'][-1]), 23)

    def test_unsupervised_collate_1D_no_zero_pad_with_stack(self):
        """Tests unsupervised_collate with zero_pad=False with stack=True 1D on waveform"""
        num = 4
        paths = [''] * num
        labels = np.ones(num)
        items = [AudioItem(path, label) for path, label in zip(paths, labels)]
        signals = [
            torch.ones((20,)),
            torch.ones((22,)),
            torch.ones((24,)),
            torch.ones((23,))
        ]

        batch = []
        for index in range(num):
            batch.append({
                'signal': signals[index],
                'label': labels[index],
                'item': items[index],
            })

        with self.assertRaises(RuntimeError):
            t_batch = unsupervised_collate(
                batch, zero_pad=False, stack=True)

    def test_unsupervised_collate_zero_pad_spectrogram_2D(self):
        """Tests unsupervised_collate with zero_pad=True on 2D spectrogram"""
        num = 4
        paths = [''] * num
        labels = np.ones(num)
        items = [AudioItem(path, label) for path, label in zip(paths, labels)]
        signals = [
            torch.ones((128, 20)),
            torch.ones((128, 22)),
            torch.ones((128, 24)),
            torch.ones((128, 23))
        ]

        batch = []
        for index in range(num):
            batch.append({
                'signal': signals[index],
                'label': labels[index],
                'item': items[index],
            })

        t_batch = unsupervised_collate(
            batch, zero_pad=True)
        self.assertEqual(t_batch['signals'].shape, (num, 128, 24))

    def test_classification_collate_zero_pad_spectrogram_2D(self):
        """Tests classification_collate with zero_pad=True on 2D spectrogram"""
        num = 4
        paths = [''] * num
        labels = np.ones(num)
        items = [AudioItem(path, label) for path, label in zip(paths, labels)]
        signals = [
            torch.ones((128, 20)),
            torch.ones((128, 22)),
            torch.ones((128, 24)),
            torch.ones((128, 23))
        ]

        batch = []
        for index in range(num):
            batch.append({
                'signal': signals[index],
                'label': labels[index],
                'item': items[index],
            })

        t_batch = classification_collate(
            batch, zero_pad=True)
        self.assertEqual(t_batch['signals'].shape, (num, 128, 24))

    def test_classification_collate_zero_pad_spectrogram_3D(self):
        """Tests classification_collate with zero_pad=True on 3D spectrogram"""
        num = 4
        paths = [''] * num
        labels = np.ones(num)
        items = [AudioItem(path, label) for path, label in zip(paths, labels)]
        signals = [
            torch.ones((2, 128, 20)),
            torch.ones((2, 128, 22)),
            torch.ones((2, 128, 24)),
            torch.ones((2, 128, 23))
        ]

        batch = []
        for index in range(num):
            batch.append({
                'signal': signals[index],
                'label': labels[index],
                'item': items[index],
            })

        t_batch = classification_collate(
            batch, zero_pad=True)
        self.assertEqual(t_batch['signals'].shape, (num, 2, 128, 24))

    def test_classification_collate_zero_pad_raw_1D(self):
        """Tests classification_collate with zero_pad=True on 1D raw waveform"""
        num = 4
        paths = [''] * num
        labels = np.ones(num)
        items = [AudioItem(path, label) for path, label in zip(paths, labels)]
        signals = [
            torch.ones(20),
            torch.ones(22),
            torch.ones(24),
            torch.ones(23)
        ]

        batch = []
        for index in range(num):
            batch.append({
                'signal': signals[index],
                'label': labels[index],
                'item': items[index],
            })

        t_batch = classification_collate(
            batch, zero_pad=True)
        self.assertEqual(t_batch['signals'].shape, (num, 24))


if __name__ == "__main__":
    unittest.main()
