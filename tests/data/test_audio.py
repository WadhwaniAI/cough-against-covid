"""Tests cac.data.audio.AudioItem"""
import os
import unittest
import torch
import numpy as np
import librosa
from cac.data.audio import AudioItem
from cac.data.utils import read_dataset_from_config
from cac.data.transforms import DataProcessor

# Transform to test spectrogram input to AudioItem
transforms_cfg = [
    {
        "name": "ToTensor",
        "params": {"device": "cpu"}
    },
    {
        "name": "Resample",
        "params": {
            "orig_freq": 44100,
            "new_freq": 16000
        }
    },
    {
        "name": "BackgroundNoise",
        "params": {
            "dataset_config": [
                {
                    "name": "esc-50",
                    "version": "default",
                    "mode": "all"
                }
            ],
            "min_noise_scale": 0.4,
            "max_noise_scale": 0.75
        }
    },
    {
        "name": "Spectrogram",
        "params": {
            "n_fft": 512,
            "win_length": 512,
            "hop_length": 160
        }
    },
    {
        "name": "MelScale",
        "params": {
            "n_mels": 64,
            "sample_rate": 16000,
            "f_min": 125,
            "f_max": 7500
        }
    },
    {
        "name": "AmplitudeToDB",
        "params": {}
    },
    {
        "name": "ToNumpy",
        "params": {}
    },
]

class AudioItemTestCase(unittest.TestCase):
    """Class to run tests on AudioItem"""
    @classmethod
    def setUpClass(cls):
        dataset_config = {
            'name': 'flusense',
            'version': 'default',
            'mode': 'val'
        }
        cls.data = read_dataset_from_config(dataset_config)
        cls.filepaths = cls.data['file']
        cls.labels = cls.data['label']
        cls.signal_transform = DataProcessor(transforms_cfg)

        dataset_config = {
            'name': 'flusense',
            'version': 'default',
            'mode': 'val'
        }
        cls.data_with_endpoints = read_dataset_from_config(dataset_config)

    def test_no_label(self):
        """Test creating AudioItem object with no label"""
        path = self.filepaths[0]
        audio = AudioItem(path=path)
        self.assertEqual(audio.info(), {'path': path})

    def test_with_label(self):
        """Test creating AudioItem object with label"""
        path = self.filepaths[0]
        label = self.labels[0]
        audio = AudioItem(path=path, label=label)
        self.assertEqual(audio.info(), {'path': path, 'label': label})

    def test_signal_numpy(self):
        """Test loading audio as numpy array"""
        audio = AudioItem(path=self.filepaths[0])
        instance = audio.load()
        signal = instance['signal']
        self.assertTrue(isinstance(signal, np.ndarray))

    def test_signal_torch(self):
        """Test loading audio as torch tensor"""
        audio = AudioItem(path=self.filepaths[0])
        instance = audio.load(as_tensor=True)
        signal = instance['signal']
        self.assertTrue(isinstance(signal, torch.Tensor))

    def test_endpoints(self):
        """Test loading audio at given start and end points in time"""
        keys = ['label', 'start', 'end']
        args = {}
        for key in keys:
            args[key] = self.data_with_endpoints[key][0]

        audio = AudioItem(path=self.data_with_endpoints['file'][0], **args)
        instance = audio.load()

        actual_duration = librosa.get_duration(
            y=instance['signal'], sr=instance['rate'])
        self.assertAlmostEqual(
            args['end'] - args['start'], actual_duration, places=4)
    
    def test_spectrogram_item(self):
        """Test spectrogram input to AudioItem"""
        audio = AudioItem(path=self.filepaths[0])
        instance = audio.load()
        signal = instance['signal']
        
        # Create spectrogram using signal transform
        transformed_signal = self.signal_transform(signal)
        path = '/tmp/temp.npy'
        np.save(path, transformed_signal)

        # Take Spectrogram Input
        spec_item = AudioItem(path = path, raw_waveform = False)
        spec_signal = spec_item.load()['signal']
        
        self.assertTrue(isinstance(spec_signal, np.ndarray))
        self.assertTrue(spec_signal.shape == transformed_signal[:, 0:-1].shape)
        self.assertTrue(np.allclose(spec_signal, transformed_signal[:, 0:-1]))

        # remove file from cache
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
