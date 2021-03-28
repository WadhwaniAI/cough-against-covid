"""Tests cac.models.classification.ClassificationModel compute_metrics"""
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
    """Class to check the metric computation of ClassificationModel"""
    @classmethod
    def setUpClass(cls):
        version = 'default.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params']['val']['fraction'] = 0.1
        cls.cfg.num_workers = 10

    def test_compute_metrics_threshold_none(self):
        """Tests no threshold specified"""
        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)

        predictions = torch.Tensor([[-1, 9], [-1.3, -1], [-2, 8]])
        targets = torch.Tensor([1, 0, 1])
        metrics = classifier.compute_epoch_metrics(predictions, targets)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['precision'], 1)

    def test_compute_metrics_threshold_given(self):
        """Tests threshold specified"""
        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)

        predictions = torch.Tensor([[-1, 9], [-1.3, -1], [-2, 8]])
        targets = torch.Tensor([1, 0, 1])
        metrics = classifier.compute_epoch_metrics(
            predictions, targets, threshold=0.6)
        self.assertEqual(metrics['threshold'], 0.6)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['precision'], 1)

    def test_compute_metrics_recall_none(self):
        """Tests minimum recall not specified"""
        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)

        predictions = torch.Tensor(
            [[-1, 9], [-1.3, -1], [-2, 8], [1.3, 2], [1.3, 1.8], [1.3, 2.1]])
        targets = torch.Tensor([1, 0, 1, 0, 1, 1])
        metrics = classifier.compute_epoch_metrics(predictions, targets)
        self.assertEqual(metrics['recall'], 1)

    def test_compute_metrics_recall_none(self):
        """Tests minimum recall not specified"""
        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = ClassificationModel(tester_cfg)

        predictions = torch.Tensor(
            [[-1, 9], [-1.3, -1], [-2, 8], [1.3, 2], [1.3, 1.8], [1.3, 2.1]])
        targets = torch.Tensor([1, 0, 1, 0, 1, 1])
        metrics = classifier.compute_epoch_metrics(
            predictions, targets, recall=0.7)
        self.assertEqual(metrics['recall'],  0.75)


if __name__ == "__main__":
    unittest.main()
