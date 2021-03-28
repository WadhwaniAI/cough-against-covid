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
        version = 'default.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params']['val']['fraction'] = 0.1
        cls.cfg.num_workers = 10

    def test_1_model_checkpoint_saving(self):
        """Tests model.save()"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)

        # remove existing checkpoints for sake of testing
        os.system('rm -rf {}/*.pth.tar'.format(tester_cfg.checkpoint_dir))

        # set epochs to be 5 in order to test saving best/regular models
        tester_cfg.model['epochs'] = 4

        # do not have to load existing checkpoints
        load_cfg = {
            'version': None,
            'epoch': -1,
            'load_best': False,
            'resume_optimizer': False
        }
        tester_cfg.model['load'] = load_cfg

        # saving after every two epochs and the best model
        save_cfg = {
            'period': 2,
            'monitor': 'precision',
            'monitor_mode': 'max'
        }
        tester_cfg.model['save'] = save_cfg

        classifier = ClassificationModel(tester_cfg)
        classifier.fit(debug=True, use_wandb=False)

        # checking both best as well as regular checkpoints
        saved_models = ['best_ckpt.pth.tar', '1_ckpt.pth.tar', '3_ckpt.pth.tar']
        for saved_model in saved_models:
            model_path = join(tester_cfg.checkpoint_dir, saved_model)
            self.assertTrue(exists(model_path))

    def test_2_model_checkpoint_loading_last_epoch(self):
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)

        load_cfg = {
            'version': 'default',
            'epoch': -1,
            'load_best': False,
            'resume_optimizer': False,
            'resume_epoch': False
        }
        tester_cfg.model['load'] = load_cfg

        classifier = ClassificationModel(tester_cfg)

        # checking if the loaded params are indeed the same as saved
        network_state = classifier.network.get_state_dict()
        load_path = get_saved_checkpoint_path(
            tester_cfg.checkpoint_dir, load_cfg['load_best'],
            load_cfg['epoch'])
        saved_state = torch.load(load_path)['network']

        for key in tqdm(network_state.keys(), desc='Testing params'):
            if key.endswith('weight'):
                network_params = network_state[key]
                saved_params = saved_state[key]
                self.assertTrue(
                    bool(torch.all(torch.eq(saved_params, network_params))))

    def test_3_model_checkpoint_loading_best_epoch(self):
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)

        # do not have to load existing checkpoints
        load_cfg = {
            'version': 'default',
            'epoch': -1,
            'load_best': True,
            'resume_optimizer': False,
            'resume_epoch': False
        }
        tester_cfg.model['load'] = load_cfg

        classifier = ClassificationModel(tester_cfg)

        # checking if the loaded params are indeed the same as saved
        network_state = classifier.network.get_state_dict()
        load_path = get_saved_checkpoint_path(
            tester_cfg.checkpoint_dir, load_cfg['load_best'],
            load_cfg['epoch'])
        self.assertIn('best_ckpt', load_path)
        saved_state = torch.load(load_path)['network']

        for key in tqdm(network_state.keys(), desc='Testing params'):
            if key.endswith('weight'):
                network_params = network_state[key]
                saved_params = saved_state[key]
                self.assertTrue(
                    bool(torch.all(torch.eq(saved_params, network_params))))


if __name__ == "__main__":
    unittest.main()
