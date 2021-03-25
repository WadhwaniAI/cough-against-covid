"""Tests cac.models.classical.ClassicalModel"""
import os
from os.path import dirname, join, exists
from copy import deepcopy
import torch
import wandb
import unittest
import random
from tqdm import tqdm
import numpy as np
from cac.config import Config
from cac.utils.logger import set_logger, color
from cac.data.dataloader import get_dataloader
from cac.models.classical import ClassicalModel


class ClassicalModelTestCase(unittest.TestCase):
    """Class to check the creation of ClassicalModel"""
    @classmethod
    def setUpClass(cls):
        version = 'defaults/classical-baseline.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params']['train']['fraction'] = 0.1
        cls.cfg.data['dataset']['params']['val']['fraction'] = 0.1
        cls.cfg.num_workers = 10

        random.seed(0)

        set_logger(join(cls.cfg.log_dir, 'train.log'))
        cls.classifier = ClassicalModel(cls.cfg)

    def test_1_model_fitting(self):
        """Test model.fit()"""
        self.classifier.fit(use_wandb=False)

    def test_2_evaluate(self):
        """Test model.evaluate()"""
        val_dataloader, _ = get_dataloader(
            self.cfg.data, 'val',
            batch_size=-1,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            drop_last=False)
        X = val_dataloader.dataset
        metrics = self.classifier.evaluate(val_dataloader)


if __name__ == "__main__":
    unittest.main()
