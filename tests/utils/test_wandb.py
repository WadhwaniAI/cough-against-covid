"""Tests cac.data.audio.AudioItem"""
import unittest
import numpy as np
import wandb
import torch
from cac.data.audio import AudioItem
from cac.data.utils import read_dataset_from_config
from cac.utils.wandb import get_audios, get_images, get_indices, get_confusion_matrix


class WandbCase(unittest.TestCase):
    """Class to run tests on wandb util functions"""
    @classmethod
    def setUpClass(cls):
        dataset_config = {
            'name': 'flusense',
            'version': 'default',
            'mode': 'val'
        }
        data_info = read_dataset_from_config(dataset_config) 
        cls.filepaths, cls.labels = data_info['file'], data_info['label']

    def test_get_indices_per_class(self):
        """Checks the function get_indices with per_class=True"""
        labels = torch.cat(
            [torch.zeros(50, dtype=int), torch.ones(50, dtype=int)]) 

        indices = get_indices(labels, per_class=True, count=4)

        self.assertEqual(len(indices), 8)
        self.assertIsInstance(indices[0], int)

        indices = np.array(indices)

        # desired output should be 4 values per class
        self.assertTrue(len(torch.nonzero(labels[indices[:4]] == 0)), 4)
        self.assertEqual(len(torch.nonzero(labels[indices[4:]] == 1)), 4)

    def test_get_indices_no_per_class(self):
        """Checks the function get_indices with per_class=False"""
        labels = torch.cat([torch.zeros(50), torch.ones(50)])

        indices = get_indices(labels, per_class=False, count=4)
        self.assertEqual(len(indices), 4)

    def test_get_confusion_matrix(self):
        """Checks the function get_confusion_matrix"""
        cm = np.random.randint(0, 10, size=(2, 2))
        classes = ['a', 'b']

        plot_as_image = get_confusion_matrix(cm, classes)
        self.assertIsInstance(plot_as_image, np.ndarray)

    def test_get_audios(self):
        """Checks the function get_audios"""
        paths = np.array(self.filepaths[:10])
        labels = torch.zeros(10)
        preds = torch.zeros(10)
        items = [AudioItem(path=path) for path in paths]

        audios = get_audios(items, preds, labels)
        self.assertEqual(len(audios), 10)
        self.assertIsInstance(audios[0], wandb.Audio)

    def test_get_images(self):
        """Checks the function get_images"""
        inputs = torch.zeros((10, 50, 50))
        labels = torch.zeros(10)
        preds = torch.zeros(10)

        images = get_images(inputs, preds, labels)
        self.assertEqual(len(images), 10)
        self.assertIsInstance(images[0], wandb.Image)


if __name__ == "__main__":
    unittest.main()
