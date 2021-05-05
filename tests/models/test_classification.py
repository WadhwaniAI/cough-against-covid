"""Tests cac.models.classification.ClassificationModel"""
import os
from os.path import dirname, join, exists
from copy import deepcopy
import torch
import wandb
import unittest
from tqdm import tqdm
import numpy as np
from torch import optim
from cac.config import Config
from cac.utils.logger import set_logger, color
from cac.models.classification import ClassificationModel


class ClassificationModelTestCase(unittest.TestCase):
    """Class to check the creation of ClassificationModel"""
    @classmethod
    def setUpClass(cls):
        version = 'default.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params']['val']['fraction'] = 0.1
        cls.cfg.num_workers = 1 if torch.cuda.is_available() else 10

    # def test_1_model_fitting(self):
    #     """Test model.fit()"""
    #     set_logger(join(self.cfg.log_dir, 'train.log'))

    #     tester_cfg = deepcopy(self.cfg)
    #     tester_cfg.model['epochs'] = 1
    #     classifier = ClassificationModel(tester_cfg)
    #     classifier.fit(debug=True, use_wandb=False)

    def test_optimizer(self):
        """Test model.fit()"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)
        self.assertIsInstance(classifier.optimizer, optim.SGD)
        self.assertIsInstance(
            classifier.scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    def test_with_frames(self):
        """Test models/lassification.py with fixed frames"""
        cfg = Config('defaults/with-frames.yml')
        cfg.data['dataset']['params']['train']['fraction'] = 0.01
        cfg.data['dataset']['params']['val']['fraction'] = 0.03
        cfg.model['batch_size'] = 4 # to make it work on small CPU machines
        cfg.num_workers = 1
        set_logger(join(cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(cfg)
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)
        classifier.fit(debug=True, use_wandb=False)

    def test_with_label_smoothing(self):
        """Test model.fit() with label smoothing"""
        tester_cfg = Config('defaults/label-smoothing-random.yml')
        set_logger(join(tester_cfg.log_dir, 'train.log'))
        tester_cfg.data['dataset']['params']['train']['fraction'] = 0.01
        tester_cfg.data['dataset']['params']['val']['fraction'] = 0.03
        tester_cfg.model['batch_size'] = 4 # to make it work on small CPU machines
        tester_cfg.num_workers = 1
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)
        classifier.fit(use_wandb=False)
        
    def test_get_unique_paths(self):
        """Tests getting unique paths with order preserved (Used in _aggregate_data())"""
        
        # input paths 
        paths = ['b', 'b', 'a', 'a', 'c', 'c', 'c', 'c']
        
        # expected unique outputs with preserved order
        exp_output = np.array(['b', 'a', 'c'])
        
        _, idx = np.unique(paths, return_index=True)
        unique_paths = np.take(paths, np.sort(idx))
        
        self.assertTrue((unique_paths == exp_output).all())

if __name__ == "__main__":
    unittest.main()
