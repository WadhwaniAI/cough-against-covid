"""Tests cac.models.classification.MultiSignalClassificationModel"""
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
from cac.models.multi_signal_classification import MultiSignalClassificationModel


class MultiSignalClassificationModelTestCase(unittest.TestCase):
    """Class to check the creation of MultiSignalClassificationModel"""
    @classmethod
    def setUpClass(cls):
        version = 'defaults/multi-signal-training.yml'
        cls.cfg = Config(version)
        cls.cfg.num_workers = 10 if torch.cuda.is_available() else 10

    def test_optimizer(self):
        """Test optimizer loading"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = MultiSignalClassificationModel(tester_cfg)
        self.assertIsInstance(classifier.optimizer, optim.SGD)
        self.assertIsInstance(
            classifier.scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    def test_loading_networks(self):
        """Test loading different networks"""
        set_logger(join(self.cfg.log_dir, 'train.log'))
        
        cough_context_cfg = Config('experiments/multi-signal-training/cough-context/naive/multi-signal-max.yml')
        cough_voice_cfg = Config('experiments/multi-signal-training/cough-voice/naive/multi-signal-min.yml')

        cough_context_model = MultiSignalClassificationModel(cough_context_cfg)
        cough_voice_model = MultiSignalClassificationModel(cough_voice_cfg)
       
        self.assertTrue(cough_context_model.network.__class__.__name__ == 'NaiveCoughContextNetwork')
        self.assertTrue(cough_voice_model.network.__class__.__name__ == 'NaiveCoughVoiceNetwork')

if __name__ == "__main__":
    unittest.main()
