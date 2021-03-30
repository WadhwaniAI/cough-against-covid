"""Tests cac.data.unsupervised_learning.UnsupervisedDataset"""
import unittest
import torch
from cac.data.unsupervised import UnsupervisedDataset
from cac.data.transforms import DataProcessor


class UnsupervisedDatasetTestCase(unittest.TestCase):
    """
    Class to run tests on UnsupervisedDataset
    
    Following configurations are tested:
    - different modes for the same dataset
    - single_dataset
    - multi_dataset
    """
    @classmethod
    def setUpClass(cls):

        cls.single_dataset_config_crowdsourced = [
            {
                'name': 'wiai-crowdsourced',
                'version': 'default-unsupervised-mini',
                'mode': 'test'
            }
        ]
        cls.single_dataset_config_facility = [
            {
                'name': 'wiai-facility',
                'version': 'default-unsupervised-mini',
                'mode': 'train'
            }
        ]
        cls.multi_dataset_config = [
            {
                'name': 'wiai-crowdsourced',
                'version': 'default-unsupervised-mini',
                'mode': 'test'
            },
            {
                'name': 'wiai-facility',
                'version': 'default-unsupervised-mini',
                'mode': 'train'
            }
        ]

        config = [
            {
                'name': 'MFCC',
                'params': {'sample_rate': 44100, 'n_mfcc': 40}
            }
        ]
        cls.spectrogram_transform = DataProcessor(config)

    def test_different_modes(self):
        """Test creating UnsupervisedDataset object for different modes"""
        test_dataset_config = {
            'name': 'wiai-crowdsourced',
            'version': 'default-unsupervised-mini',
            'mode': 'test'
        }
        train_dataset_config = {
            'name': 'wiai-crowdsourced',
            'version': 'default-unsupervised-mini',
            'mode': 'train'
        }
        test_dataset = UnsupervisedDataset(dataset_config=[test_dataset_config])
        train_dataset = UnsupervisedDataset(dataset_config=[train_dataset_config])

        self.assertTrue(len(test_dataset.items) != len(train_dataset.items))

    def test_single_dataset_crowdsourced(self):
        """Checks single dataset for crowdsourced"""
        dataset = UnsupervisedDataset(
            dataset_config=self.single_dataset_config_crowdsourced,
            signal_transform=self.spectrogram_transform)

        instance = dataset[0]

        self.assertTrue(isinstance(instance['signal'], torch.Tensor))
        self.assertTrue(instance['label']['dataset-name'] == 'wiai-crowdsourced')

    def test_single_dataset_facility(self):
        """Checks single dataset for facility"""
        dataset = UnsupervisedDataset(
            dataset_config=self.single_dataset_config_facility,
            signal_transform=self.spectrogram_transform)

        instance = dataset[0]

        self.assertTrue(isinstance(instance['signal'], torch.Tensor))
        self.assertTrue(instance['label']['dataset-name'] == 'wiai-facility')

    def test_multi_dataset(self):
        """Checks multi dataset"""
        dataset = UnsupervisedDataset(dataset_config=self.multi_dataset_config,
            signal_transform=self.spectrogram_transform)

        instance_crowdsourced = dataset[0]
        self.assertTrue(instance_crowdsourced['label']['dataset-name'] == 'wiai-crowdsourced')

        instance_facility = dataset[-1]
        self.assertTrue(instance_facility['label']['dataset-name'] == 'wiai-facility')


if __name__ == "__main__":
    unittest.main()
