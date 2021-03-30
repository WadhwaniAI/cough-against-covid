"""Tests cac.data.base.BaseDataset"""
import unittest
from os.path import join
import torch
import numpy as np
from cac.data.audio import AudioItem
from cac.data.base import BaseDataset
from cac.data.utils import read_dataset_from_config


class BaseDatasetTestCase(unittest.TestCase):
    """Class to run tests on BaseDataset"""
    def test_single_dataset_single_task(self):
        """Test creating BaseDataset object for one dataset with a single task"""
        dataset_config = {
            'name': 'flusense',
            'version': 'segmented-v1.0',
            'mode': 'val'
        }
        dataset = BaseDataset([dataset_config])

        data_info = read_dataset_from_config(dataset_config)
        filepaths = data_info['file']

        self.assertEqual(len(filepaths), len(dataset.items))
        self.assertTrue(isinstance(dataset.items[0], AudioItem))

    def test_fraction(self):
        """Test creating BaseDataset object and using fraction < 1"""
        dataset_config = {
            'name': 'flusense',
            'version': 'segmented-v1.0',
            'mode': 'val'
        }
        fraction = 0.5
        dataset = BaseDataset([dataset_config], fraction=fraction)
        data_info = read_dataset_from_config(dataset_config)
        filepaths = data_info['file']
        self.assertEqual(int(len(filepaths) * fraction), len(dataset.items))

    def test_different_modes(self):
        """Test creating BaseDataset object for different modes"""
        val_dataset_config = {
            'name': 'flusense',
            'version': 'segmented-v1.0',
            'mode': 'val'
        }
        train_dataset_config = {
            'name': 'flusense',
            'version': 'segmented-v1.0',
            'mode': 'train'
        }
        val_dataset = BaseDataset([val_dataset_config])
        train_dataset = BaseDataset([train_dataset_config])

        self.assertTrue(len(val_dataset.items) != len(train_dataset.items))

    def test_multiple_datasets_single_task(self):
        """Test creating BaseDataset object for multiple datasets with a single task"""
        dataset_config = [
            {
                'name': 'flusense',
                'version': 'segmented-v1.0',
                'mode': 'train'
            },
            {
                'name': 'flusense',
                'version': 'segmented-v1.0',
                'mode': 'val'
            }
        ]
        dataset = BaseDataset(dataset_config)

        num = 0

        for dataset_config in dataset_config:
            data_info = read_dataset_from_config(dataset_config)
            filepaths = data_info['file']
            num += len(filepaths)

        self.assertEqual(num, len(dataset.items))
        self.assertTrue(isinstance(dataset.items[0], AudioItem))

    def test_create_frames(self):
        """Test create_frames"""
        dataset_config = {
            'name': 'flusense',
            'version': 'segmented-v1.0',
            'mode': 'val'
        }
        dataset = BaseDataset(
            [dataset_config], as_frames=True, frame_length=1000, min_length=500, hop_length=500
        )

        data_info = read_dataset_from_config(dataset_config)
        filepaths = data_info['file']
        labels = data_info['label']
        start = np.array(data_info['start'])
        end = np.array(data_info['end'])
        diff = end - start

        # file longer than frame length with last frame length < min_length
        self.assertEqual(filepaths[30],
            '/data/flusense/processed/audio/SF-srscAiK0_0_000-10_000.wav')
        items = dataset._create_frames(filepaths[30], labels[30], start[30], end[30])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].end - items[0].start, 1)

        # file longer than frame length with last frame length > min_length
        self.assertEqual(filepaths[1],
            '/data/flusense/processed/audio/FpuF3uuV1U0_7_000-17_000.wav')
        items = dataset._create_frames(filepaths[1], labels[1], start[1], end[1])
        self.assertEqual(len(items), 4)
        for item in items:
            self.assertEqual(item.end - item.start, 1)

        # file shorter than frame length
        self.assertEqual(filepaths[10],
            '/data/flusense/processed/audio/cl4hzarjGhk_0_000-10_000.wav')
        items = dataset._create_frames(filepaths[10], labels[10], start[10], end[10])
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].end - items[0].start < 1)


if __name__ == "__main__":
    unittest.main()
