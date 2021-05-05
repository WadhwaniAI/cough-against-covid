"""Tests cac.models.classification.ClassificationModel evaluation"""
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
from cac.data.dataloader import get_dataloader
from cac.models.classification import ClassificationModel
from cac.models.utils import get_saved_checkpoint_path


class ClassificationModelEvaluationTestCase(unittest.TestCase):
    """Class to check the evaluation of ClassificationModel"""
    @classmethod
    def setUpClass(cls):
        version = 'default.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params']['train']['fraction'] = 0.01
        cls.cfg.data['dataset']['params']['val']['fraction'] = 0.03
        cls.cfg.num_workers = 1 if torch.cuda.is_available() else 10

    def test_1_model_fitting(self):
        """Test model.fit()"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)
        classifier.fit(debug=True, use_wandb=False)

    def test_2_evaluate(self):
        """Test model.evaluate()"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['load']['version'] = 'default'
        tester_cfg.model['load']['load_best'] = True
        kwargs = {'threshold': 0.5}
        model = ClassificationModel(tester_cfg)
        dataloader, _ = get_dataloader(
            tester_cfg.data, 'val',
            tester_cfg.model['batch_size'],
            num_workers=4,
            shuffle=False,
            drop_last=False)
        model.evaluate(dataloader, 'val', False, **kwargs)


if __name__ == "__main__":
    unittest.main()
