"""Tests cac.models.dimensionality_reduction.DimensionalityReductionModel"""
import os
from os.path import dirname, join, exists
from copy import deepcopy
import torch
import wandb
import unittest
from tqdm import tqdm
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU
from cac.config import Config
from cac.utils.logger import set_logger, color
from cac.models.dimensionality_reduction import DimensionalityReductionModel
from cac.models.utils import get_saved_checkpoint_path


class DimensionalityReductionModelTestCase(unittest.TestCase):
    """Class to check the creation of DimensionalityReductionModel"""
    @classmethod
    def setUpClass(cls):
        version = 'defaults/unsupervised.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params'] = {
            'all': {
                'fraction': 0.01
            }
        }
        cls.cfg.num_workers = 10

    def test_pca_model_fitting(self):
        """Test model.fit()"""
        set_logger(join(self.cfg.log_dir, 'unsupervised.log'))

        tester_cfg = deepcopy(self.cfg)
        reduction_model = DimensionalityReductionModel(tester_cfg)
        data = reduction_model.fit(return_predictions=True)

        X, Z, Y = data['input'], data['latent'], data['labels']

        self.assertEqual(Z.shape[-1], 2)
        self.assertEqual(X.shape[0], Z.shape[0])
        self.assertEqual(Z.shape[0], len(Y))
        self.assertTrue(Z.shape[-1] <= X.shape[-1])

    def test_tsne_model_fitting(self):
        """Test model.fit()"""
        set_logger(join(self.cfg.log_dir, 'unsupervised.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.__dict__['model']['method']['name'] = 'TSNE'
        reduction_model = DimensionalityReductionModel(tester_cfg)
        data = reduction_model.fit(return_predictions=True)

        X, Z, Y = data['input'], data['latent'], data['labels']

        self.assertEqual(Z.shape[-1], 2)
        self.assertEqual(X.shape[0], Z.shape[0])
        self.assertEqual(Z.shape[0], len(Y))
        self.assertTrue(Z.shape[-1] <= X.shape[-1])


if __name__ == "__main__":
    unittest.main()
