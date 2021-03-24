"""Tests cac.data.classification.ClassificationDataset"""
import unittest
import torch
from cac.data.utils import read_dataset_from_config


class ClassificationDatasetTestCase(unittest.TestCase):
    """Class to run tests on ClassificationDataset

    Tests the following cases
    - Flusense dataset with default config
    - Flusense dataset with split audio segments config
    - Stonybrook dataset with default config
    """

    def test_1_flusense_default_config(self):
        flusense_default_cfg = {
            'name': 'flusense',
            'version': 'default',
            'mode': 'val'
        }
        dataset_info = read_dataset_from_config(flusense_default_cfg)

        self.assertIn('file', dataset_info.keys())
        self.assertIn('label', dataset_info.keys())

    def test_2_flusense_segmented_config(self):
        flusense_segmented_cfg = {
            'name': 'flusense',
            'version': 'segmented-v1.0',
            'mode': 'val'
        }
        dataset_info = read_dataset_from_config(flusense_segmented_cfg)

        self.assertIn('file', dataset_info.keys())
        self.assertIn('label', dataset_info.keys())
        self.assertIn('start', dataset_info.keys())
        self.assertIn('end', dataset_info.keys())


if __name__ == "__main__":
    unittest.main()
