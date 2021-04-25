"""Tests cac.data.classification.MultiSignalClassificationDataset"""
import unittest
import torch
from cac.config import Config
from cac.data.multi_signal_classification import MultiSignalClassificationDataset
from cac.data.transforms import DataProcessor, ClassificationAnnotationTransform


class MultiSignalClassificationDatasetTestCase(unittest.TestCase):
    """
    Class to run tests on MultiSignalClassificationDataset
    
    Following configurations are tested:
    - different modes
    - multiple datasets
    - contextual data check
    """
    @classmethod
    def setUpClass(cls):
        cfg = Config('defaults/multi-signal-training.yml')
        
        cls.single_data_config = cfg.data.copy()

        cls.multiple_data_config = cfg.data.copy()
        cls.multiple_data_config['data'].append(cls.multiple_data_config['data'][0])

        cls.features = ['enroll_patient_age']
        cls.attributes_file = '/data/wiai-facility/processed/attributes_context_processed.csv'
        cls.single_dataset_dataset_params = {
            'cfg' : cls.single_data_config,
            'features' : cls.features,
            'attribute_file': cls.attributes_file,
        }  
        cls.multiple_dataset_dataset_params = {
            'cfg' : cls.multiple_data_config,
            'features' : cls.features,
            'attribute_file': cls.attributes_file,
        }  

    def test_different_modes(self):
        """Test creating ContextClassificationDataset object for different modes"""
        self.single_dataset_dataset_params['mode'] = 'val'
        val_dataset = MultiSignalClassificationDataset(**self.single_dataset_dataset_params)
        self.single_dataset_dataset_params['mode'] = 'train'
        test_dataset = MultiSignalClassificationDataset(**self.single_dataset_dataset_params)
        self.assertTrue(len(val_dataset.items) != len(test_dataset.items))

    def test_loading_multiple_datasets(self):
        """Checks if multiple datasets can be loaded"""
        self.multiple_dataset_dataset_params['mode'] = 'val'
        multiple_dataset = MultiSignalClassificationDataset(**self.multiple_dataset_dataset_params)
        self.assertTrue(len(multiple_dataset[0]['signal']) == 2)

    def test_loading_context_data(self):
        """Checks if multiple datasets can be loaded"""
        self.single_dataset_dataset_params['mode'] = 'val'
        dataset = MultiSignalClassificationDataset(**self.single_dataset_dataset_params)
        self.assertTrue(dataset[0]['context-signal'].shape[1] == len(self.features))

if __name__ == "__main__":
    unittest.main()
