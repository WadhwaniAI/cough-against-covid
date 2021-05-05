"""Tests cac.models.optimizer.Optimizer"""
import torch
import unittest
from cac.networks.nn import NeuralNetwork
from cac.optimizer import optimizer_factory


class OptimizerTestCase(unittest.TestCase):
    """Class to check the creation of Optimizer"""
    @classmethod
    def setUpClass(cls):
        cfg = [
            {
                'name': "Conv2d",
                'params': {
                    "in_channels": 1,
                    "out_channels": 16,
                    "kernel_size": [3, 7]
                }
            },
            {
                'name': "BatchNorm2d",
                'params': {
                    "num_features": 16
                }
            },
            {
                'name': "LeakyReLU",
                'params': {}
            }
        ]
        cls.network = NeuralNetwork(cfg)

    def test_adam(self):
        """Test creation of a Adam optmizer"""
        optimizer_name = 'Adam'
        optimizer_args = {
            'params': self.network.parameters(),
            'lr': 0.0003,
            'weight_decay': 0.0005
        }

        optimizer = optimizer_factory.create(optimizer_name, **optimizer_args)
        self.assertTrue(optimizer_name in optimizer.__doc__)

    def test_sgd(self):
        """Test creation of a SGD optmizer"""
        optimizer_name = 'SGD'
        optimizer_args = {
            'params': self.network.parameters(),
            'lr': 0.0003,
            'weight_decay': 0.0005
        }

        optimizer = optimizer_factory.create(optimizer_name, **optimizer_args)
        self.assertTrue(optimizer_name in optimizer.__doc__)


if __name__ == "__main__":
    unittest.main()
