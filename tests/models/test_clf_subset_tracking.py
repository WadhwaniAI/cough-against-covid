"""Tests cac.models.classification.ClassificationModel subset tracking"""
import os
from os.path import dirname, join, exists
from copy import deepcopy
import torch
import wandb
import unittest
from tqdm import tqdm
import numpy as np
from torch import optim
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU
from cac.config import Config
from cac.utils.logger import set_logger, color
from cac.models.classification import ClassificationModel


class ClassificationModelTestCase(unittest.TestCase):
    """Class to check the subset tracking of ClassificationModel"""
    @classmethod
    def setUpClass(cls):
        version = 'default.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params']['train']['fraction'] = 0.01
        cls.cfg.data['dataset']['params']['val']['fraction'] = 0.03
        cls.cfg.num_workers = 1 if torch.cuda.is_available() else 10

    def test_subset_tracking(self):
        """Test subset_tracker in base.py"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        tester_cfg.data['dataset']['config'] = [
            {
                'name': 'wiai-facility',
                'version': 'v1.2',
            }
        ]
        tester_cfg.model['subset_tracker'] = {
            'val': [
                {
                    'name': 'wiai-facility',
                    'version': 'v4.0',
                    'mode': 'val-nmch'
                },
                {
                    'name': 'wiai-facility',
                    'version': 'v4.0',
                    'mode': 'val-mzf'
                }

            ]
        }
        classifier = ClassificationModel(tester_cfg)
        classifier.fit(debug=True, use_wandb=False)


if __name__ == "__main__":
    unittest.main()
