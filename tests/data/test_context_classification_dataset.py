"""Tests cac.data.classification.ContextClassificationDataset"""
import unittest
import torch
from cac.data.context_classification import ContextClassificationDataset
from cac.data.transforms import DataProcessor, ClassificationAnnotationTransform


class ContextClassificationDatasetTestCase(unittest.TestCase):
    """
    Class to run tests on ContextClassificationDataset
    
    Following configurations are tested:
    - single_dataset + binary + no transform
    - single_dataset + binary + transform
    - multi_dataset + multi_classes + transform
    """
    @classmethod
    def setUpClass(cls):
        cls.single_dataset_config = [
            {
                'name': 'wiai-facility',
                'version': 'v9.8',
                'mode': 'val'
            }
        ]
        cls.multi_dataset_config = [
            {
                'name': 'wiai-facility',
                'version': 'v9.8',
                'mode': 'val'
            },
            {
                'name': 'wiai-facility',
                'version': 'v9.8',
                'mode': 'test'
            }
        ]
        cls.features = ['enroll_patient_age']
        cls.attributes_file = '/data/wiai-facility/processed/attributes_context_processed.csv'

    def test_different_modes(self):
        """Test creating ContextClassificationDataset object for different modes"""
        val_dataset_config = {
            'name': 'wiai-facility',
            'version': 'default-clf',
            'mode': 'test'
        }
        train_dataset_config = {
            'name': 'wiai-facility',
            'version': 'default-clf',
            'mode': 'val'
        }
        
        val_dataset = ContextClassificationDataset(dataset_config = [val_dataset_config], features = self.features,
                                                    attributes_file = self.attributes_file)
        train_dataset = ContextClassificationDataset(dataset_config = [train_dataset_config], features = self.features,
                                                    attributes_file = self.attributes_file)
        self.assertTrue(len(val_dataset.items) != len(train_dataset.items))


    def test_loading_multiple_datasets(self):
        """Checks if multiple datasets can be loaded"""
        dataset = ContextClassificationDataset(dataset_config = self.multi_dataset_config, features = self.features,
                                                    attributes_file = self.attributes_file)
        val_dataset = ContextClassificationDataset(dataset_config = self.single_dataset_config, features = self.features,
                                                    attributes_file = self.attributes_file)
        self.assertTrue(len(val_dataset.items) < len(dataset.items))

    def test_single_dataset_binary_class_no_transform(self):
        """Checks single dataset for binary classification task using no transform"""
        dataset = ContextClassificationDataset(dataset_config = self.single_dataset_config, features = self.features,
                                                    attributes_file = self.attributes_file)
        
        instance = dataset[0]
        self.assertEqual(
            instance['item'].path, '/data/wiai-facility/processed/audio/patient_ebbc8e842a3c58e3557fa95730b28fb036d49116_20200806_142500_cough_1.wav')
        self.assertTrue(isinstance(instance['signal'], torch.Tensor))
        self.assertEqual(instance['label'], [])

        instance = dataset[5]
        self.assertEqual(
            instance['item'].path, '/data/wiai-facility/processed/audio/patient_4eebd153d51051a4bbcc9ca1d07b7239551c8485_20200919_172912_cough_1.wav')
        self.assertTrue(isinstance(instance['signal'], torch.Tensor))
        self.assertEqual(instance['label'], ['covid'])


if __name__ == "__main__":
    unittest.main()
