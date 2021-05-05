"""Tests cac.models.nn.NeuralNetwork"""
import torch
import unittest
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU
from numpy.testing import assert_array_equal
from cac.config import Config
from cac.networks.nn import NeuralNetwork


class NeuralNetworkTestCase(unittest.TestCase):
    """Class to check the creation of NeuralNetwork"""
    def test_cnn_creation(self):
        """Test creation of a CNN using NeuralNetwork"""
        cfg = [
            {
                "name": "Conv2d",
                "params": {
                    "in_channels": 1,
                    "out_channels": 16,
                    "kernel_size": [3, 7]
                }
            },
            {
                "name": "BatchNorm2d",
                "params": {
                    "num_features": 16
                }
            },
            {
                "name": "LeakyReLU",
                "params": {}
            }
        ]

        network = NeuralNetwork(cfg)
        self.assertEqual(len(cfg), len(network.blocks))

        # test Conv2d
        self.assertIsInstance(network.blocks[0], Conv2d)
        self.assertEqual(network.blocks[0].in_channels, 1)
        self.assertEqual(network.blocks[0].out_channels, 16)
        self.assertEqual(network.blocks[0].kernel_size, [3, 7])

        # test BatchNorm2d
        self.assertIsInstance(network.blocks[1], BatchNorm2d)
        self.assertEqual(network.blocks[1].num_features, 16)

        # test LeakyReLU
        self.assertIsInstance(network.blocks[2], LeakyReLU)

    def test_init(self):
        """Test weight initialization of a NeuralNetwork"""

        network_config = [
          {
            "name": "Conv2d",
            "params": {
                "in_channels": 1,
                "out_channels": 64,
                "padding": [3, 1],
                "kernel_size": [7, 3]
            },
            "requires_grad": False
          },
          {
            "name": "BatchNorm2d",
            "params": {
                "num_features": 64
            },
            "requires_grad": False
          },
          {
            "name": "ReLU",
            "params": {}
          },
          {
            "name": "AdaptiveAvgPool2d",
            "params": {
                "output_size": [1, 1]
            }
          },
          {
            "name": "Flatten",
            "params": {}
          },
          {
            "name": "Linear",
            "params": {
                "in_features": 64,
                "out_features": 32
            }
          },
          {
            "name": "ReLU",
            "params": {}
          },
          {
            "name": "Linear",
            "params": {
                "in_features": 32,
                "out_features": 2
            }
          }
        ]

        network_init = {
            "weight": {
                "name": "kaiming_uniform",
                "params": {
                    "mode": "fan_out",
                    "nonlinearity": "relu"
                }
            },
            "bias": {
                "name": "zeros"
            },
            "bn_weight": {
                "name": "constant",
                "params": {
                    "val": 1
                }
            },
            "bn_bias": {
                "name": "constant",
                "params": {
                    "val": 0
                }
            }
        }

        network_init['weight']['name'] = 'constant'
        network_init['weight']['params'] = {'val': 0}
        network_init['bias']['name'] = 'constant'
        network_init['bias']['params'] = {'val': 0}
        network_init['bn_weight']['name'] = 'constant'
        network_init['bn_weight']['params'] = {'val': 0}
        network_init['bn_bias']['name'] = 'constant'
        network_init['bn_bias']['params'] = {'val': 0}
        network = NeuralNetwork(network_config, network_init)

        for m in network.modules():
            if isinstance(m, BatchNorm2d) or isinstance(m, Conv2d):
                assert_array_equal(m.weight.detach().numpy(), 0)
                assert_array_equal(m.bias.detach().numpy(), 0)

        config = Config('default.yml')
        network_config = config.network['params']['config']
        network = NeuralNetwork(network_config)

        for m in network.modules():
            if isinstance(m, BatchNorm2d):
                assert_array_equal(m.weight.detach().numpy(), 1)
                assert_array_equal(m.bias.detach().numpy(), 0)

    def test_freeze_layers(self):
        """Test freezing layers of a network"""
        config = [
          {
            "name": "Conv2d",
            "params": {
                "in_channels": 1,
                "out_channels": 64,
                "padding": [3, 1],
                "kernel_size": [7, 3]
            },
            "requires_grad": False
          },
          {
            "name": "BatchNorm2d",
            "params": {
                "num_features": 64
            },
            "requires_grad": False
          },
          {
            "name": "ReLU",
            "params": {}
          },
          {
            "name": "AdaptiveAvgPool2d",
            "params": {
                "output_size": [1, 1]
            }
          },
          {
            "name": "Flatten",
            "params": {}
          },
          {
            "name": "Linear",
            "params": {
                "in_features": 64,
                "out_features": 32
            }
          },
          {
            "name": "ReLU",
            "params": {}
          },
          {
            "name": "Linear",
            "params": {
                "in_features": 32,
                "out_features": 2
            }
          }
        ]

        network = NeuralNetwork(config)
        network.freeze_layers()
        for name, param in network.named_parameters():
            if '0_' in name or '1_' in name:
                self.assertEqual(param.requires_grad, False)
            else:
                self.assertEqual(param.requires_grad, True)
        dummy = torch.zeros(4, 1, 128, 40)
        out = network(dummy)
        self.assertEqual(
            out.shape, torch.Size([4, network.blocks[-1].out_features]))

    def test_freeze_backbone_layers(self):
        """Test freezing layers of backbone in a network"""

        config = [
          {
            "name": "resnet18",
            "params": {
                "in_channels": 1,
                "pretrained": True
            },
            "requires_grad": False
          },
          {
            "name": "AdaptiveAvgPool2d",
            "params": {
                "output_size": [1, 1]
            }
          },
          {
            "name": "Flatten",
            "params": {}
          },
          {
            "name": "Linear",
            "params": {
                "in_features": 512,
                "out_features": 32
            }
          },
          {
            "name": "ReLU",
            "params": {}
          },
          {
            "name": "Linear",
            "params": {
                "in_features": 32,
                "out_features": 2
            }
          }
        ]

        network = NeuralNetwork(config)
        network.freeze_layers()
        for name, param in network.named_parameters():
            if '0_' in name or '1_' in name:
                self.assertEqual(param.requires_grad, False)
            else:
                self.assertEqual(param.requires_grad, True)
        dummy = torch.zeros(4, 1, 128, 40)
        out = network(dummy)
        self.assertEqual(
            out.shape, torch.Size([4, network.blocks[-1].out_features]))

    def test_cnn_forward(self):
        """Test forward pass of a CNN using NeuralNetwork"""
        config = Config('default.yml')
        network_config = config.network['params']['config']
        network = NeuralNetwork(network_config)
        dummy = torch.zeros(4, 1, 128, 40)
        out = network(dummy)
        self.assertEqual(
            out.shape, torch.Size([4, network.blocks[-1].out_features]))

    def test_resnet_backbone_with_layer(self):
        """Test using resnet backbone in combination with other layers"""
        cfg = [
            {
                "name": "resnet18",
                "params": {
                    "in_channels": 1,
                    "pretrained": True,
                }
            },
            {
                "name": "AdaptiveAvgPool2d",
                "params": {
                    "output_size": (1, 1)
                }
            },
            {
                "name": "Flatten",
                "params": {}
            },
            {
                "name": "Linear",
                "params": {
                    'in_features': 512,
                    'out_features': 2
                }
            }
        ]

        network = NeuralNetwork(cfg)
        dummy = torch.zeros(4, 1, 250, 250)
        out = network(dummy)
        self.assertEqual(out.shape, torch.Size([4, 2]))

    def test_vgg_backbone_with_layer(self):
        """Test using vgg backbone in combination with other layers"""
        cfg = [
            {
                "name": "vgg19",
                "params": {
                    "in_channels": 1,
                    "pretrained": True,
                }
            },
            {
                "name": "AdaptiveAvgPool2d",
                "params": {
                    "output_size": (1, 1)
                }
            },
            {
                "name": "Flatten",
                "params": {}
            },
            {
                "name": "Linear",
                "params": {
                    'in_features': 512,
                    'out_features': 2
                }
            }
        ]

        network = NeuralNetwork(cfg)
        dummy = torch.zeros(4, 1, 250, 250)
        out = network(dummy)
        self.assertEqual(out.shape, torch.Size([4, 2]))


if __name__ == "__main__":
    unittest.main()
