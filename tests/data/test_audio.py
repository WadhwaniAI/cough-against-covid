"""Tests cac.data.audio.AudioItem"""
import unittest
import torch
import numpy as np
import librosa
from cac.data.audio import AudioItem
from cac.data.utils import read_dataset_from_config


class AudioItemTestCase(unittest.TestCase):
    """Class to run tests on AudioItem"""
    @classmethod
    def setUpClass(cls):
        dataset_config = {
            'name': 'flusense',
            'version': 'segmented-v1.0',
            'mode': 'val'
        }
        cls.data = read_dataset_from_config(dataset_config)
        cls.filepaths = cls.data['file']
        cls.labels = cls.data['label']

        dataset_config = {
            'name': 'flusense',
            'version': 'segmented-v1.0',
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


if __name__ == "__main__":
    unittest.main()
