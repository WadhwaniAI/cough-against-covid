"""Tests cac.models.classification.ClassificationModel checkpoint load/save"""
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
from cac.models.utils import get_saved_checkpoint_path


class ClassificationModelTestCase(unittest.TestCase):
    """Class to check the checkpoint saving/loading of ClassificationModel"""
    @classmethod
    def setUpClass(cls):
        version = 'default-file-agg.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params']['val']['fraction'] = 0.1
        cls.cfg.num_workers = 10

    def test_no_file_aggregation(self):
        """Tests model without file-level aggregation"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        tester_cfg.model['eval']['aggregate'] = {
            'train': {},
            'val': {}
        }

        classifier = ClassificationModel(tester_cfg)
        classifier.fit(debug=True, use_wandb=False)

    def test_with_file_aggregation(self):
        """Tests model with file-level aggregation"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)
        classifier.fit(debug=True, use_wandb=False)


if __name__ == "__main__":
    unittest.main()
