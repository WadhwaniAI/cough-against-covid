"""Tests cac.data.classification.ClassificationDataset"""
import unittest
import torch
from cac.data.classification import ClassificationDataset
from cac.data.transforms import DataProcessor, ClassificationAnnotationTransform


class ClassificationDatasetTestCase(unittest.TestCase):
    """
    Class to run tests on ClassificationDataset
    
    Following configurations are tested:
    - single_dataset + binary + no transform
    - single_dataset + binary + transform
    - multi_dataset + multi_classes + transform
    """
    @classmethod
    def setUpClass(cls):
        binary_classes = ['cough']
        multi_classes = ['sneeze', 'cough', 'bark']
        cls.binary_transform = ClassificationAnnotationTransform(binary_classes)
        cls.multi_transform = ClassificationAnnotationTransform(multi_classes)

        cls.single_dataset_config = [
            {
                'name': 'flusense',
                'version': 'default',
                'mode': 'val'
            }
        ]
        cls.multi_dataset_config = [
            {
                'name': 'flusense',
                'version': 'default',
                'mode': 'train'
            },
            {
                'name': 'flusense',
                'version': 'default',
                'mode': 'val'
            }
        ]

        cls.n_fft = 440
        config = [
            {
                'name': 'Spectrogram',
                'params': {'n_fft': cls.n_fft, 'win_length': None, 'hop_length': None}
            }
        ]
        cls.spectrogram_transform = DataProcessor(config)

    def test_different_modes(self):
        """Test creating ClassificationDataset object for different modes"""
        val_dataset_config = {
            'name': 'flusense',
            'version': 'default',
            'mode': 'val'
        }
        train_dataset_config = {
            'name': 'flusense',
            'version': 'default',
            'mode': 'train'
        }
        val_dataset = ClassificationDataset([val_dataset_config])
        train_dataset = ClassificationDataset([train_dataset_config])

        self.assertTrue(len(val_dataset.items) != len(train_dataset.items))

    def test_single_dataset_binary_class_no_transform(self):
        """Checks single dataset for binary classification task using no transform"""
        dataset = ClassificationDataset(self.single_dataset_config)

        instance = dataset[0]
        self.assertEqual(instance['item'].path, '/data/flusense/processed/audio/0oUkEze_kmo.wav')
        self.assertTrue(isinstance(instance['signal'], torch.Tensor))
        self.assertEqual(instance['label'], ['cough'])

        instance = dataset[3]
        self.assertEqual(
            instance['item'].path, '/data/flusense/processed/audio/MPUShjrdzow_30_000-40_000.wav')
        self.assertTrue(isinstance(instance['signal'], torch.Tensor))
        self.assertEqual(instance['label'], [])

    def test_single_dataset_binary_class_with_transform(self):
        """Checks single dataset for binary classification task using target transform"""
        dataset = ClassificationDataset(
            self.single_dataset_config,
            target_transform=self.binary_transform)

        instance = dataset[0]
        self.assertEqual(instance['label'], 1)

        instance = dataset[3]
        self.assertEqual(instance['label'], 0)

    def test_multi_dataset_multi_class_with_transform(self):
        """Checks multi dataset for multiclass classification task using target transform"""
        dataset = ClassificationDataset(
            self.multi_dataset_config,
            target_transform=self.multi_transform)

        instance = dataset[0]
        self.assertEqual(
            instance['item'].path,
            '/data/flusense/processed/audio/OwAUGABGrqk.wav')
        self.assertEqual(instance['label'], 2)

    def test_single_dataset_binary_class_with_signal_transform(self):
        """Checks single dataset for binary classification task with signal transform"""
        dataset = ClassificationDataset(
            self.single_dataset_config,
            signal_transform=self.spectrogram_transform,
            target_transform=self.binary_transform)

        instance = dataset[1]

        frame_vector_length = self.n_fft // 2 + 1
        num_frames = 2005
        self.assertEqual(instance['signal'].shape, (frame_vector_length, num_frames))
        self.assertEqual(instance['label'], 0)


if __name__ == "__main__":
    unittest.main()
