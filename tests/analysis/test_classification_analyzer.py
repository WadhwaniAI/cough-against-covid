"""Tests cac.analysis.classification.ClassificationAnalyzer"""
import os
from os.path import dirname, join, exists
from copy import deepcopy
import torch
import wandb
import unittest
from tqdm import tqdm

from cac.config import Config
from cac.data.dataloader import get_dataloader
from cac.utils.logger import set_logger, color
from cac.analysis.classification import ClassificationAnalyzer


class ClassificationAnalyzerTestCase(unittest.TestCase):
    """Class to check the creation of ClassificationModel"""
    @classmethod
    def setUpClass(cls):
        cfg = 'experiments/covid-detection/v9_4_cough_adam_1e-4.yml'
        cls.config = Config(cfg)
        print(f"WARNING: If you have not run expt with config {cfg}, tests will fail. Please run\
            `python training/train.py -v {cfg}` before running this test.")
        cls.data_loader, _ = get_dataloader(
            cls.config.data, 'val', 10,
            num_workers=4, shuffle=False, drop_last=False)
        cls.common_analyzer = ClassificationAnalyzer(
            cls.config, checkpoint=-1,
            load_best=False, debug=True)

    def test_analyzer_init_with_last_checkpoint(self):
        """Tests initiating analyzer with last checkpoint"""
        analyzer = ClassificationAnalyzer(
            self.config, checkpoint=-1,
            load_best=False, debug=True)

    def test_analyzer_init_with_best_checkpoint(self):
        """Tests initiating analyzer with best checkpoint"""
        analyzer = ClassificationAnalyzer(
            self.config, checkpoint=-1,
            load_best=True, debug=True)

    def test_analyzer_features_with_last_layer(self):
        """Tests analyzer to compute last layer features"""
        results = self.common_analyzer.compute_features(
            self.data_loader,
            last_layer_index=-1)

        self.assertIn('features', results)
        self.assertIn('attributes', results)
        self.assertIn('unique_id', results['attributes'][0])
        self.assertIn('audio_type', results['attributes'][0])
        self.assertEqual(len(self.data_loader.dataset), results['features'].shape[0])
        self.assertEqual(len(results['features'].shape), 2)

    def test_analyzer_features_post_compression(self):
        """Tests analyzer to compute last layer features"""
        results = self.common_analyzer.compute_features(
            self.data_loader,
            last_layer_index=-1)

        method_cfg = {
            'name': 'PCA',
            'params': {'n_components': 2}
        }
        X = results['features']
        Z = self.common_analyzer.compute_embeddings(method_cfg, X)

        self.assertEqual(len(self.data_loader.dataset), Z.shape[0])
        self.assertEqual(Z.shape, (len(self.data_loader.dataset), 2))


if __name__ == "__main__":
    unittest.main()
