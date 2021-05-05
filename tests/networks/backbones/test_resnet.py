"""Tests cac.models.backbones.resnet"""
import torch
import torch.nn as nn
import unittest
from cac.networks.backbones.resnet import resnet18, resnet34, resnet50, \
    resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,\
    wide_resnet50_2, wide_resnet101_2


class ResnetTestCase(unittest.TestCase):
    """Class to check the Resnet-based backbones"""
    def test_resnet18(self):
        """Test resnet-18"""
        net = resnet18(pretrained=True)

    def test_resnet34(self):
        """Test resnet-34"""
        net = resnet34(pretrained=True)

    def test_resnet50(self):
        """Test resnet-50"""
        net = resnet50(pretrained=True)

    def test_resnet101(self):
        """Test resnet-101"""
        net = resnet101(pretrained=True)

    def test_resnet152(self):
        """Test resnet-152"""
        net = resnet152(pretrained=True)

    def test_resnext50_32x4d(self):
        """Test resnext50_32x4d"""
        net = resnext50_32x4d(pretrained=True)

    def test_resnext101_32x8d(self):
        """Test resnext101_32x8d"""
        net = resnext101_32x8d(pretrained=True)

    def test_wide_resnet50_2(self):
        """Test wide_resnet50_2"""
        net = wide_resnet50_2(pretrained=True)

    def test_wide_resnet101_2(self):
        """Test wide_resnet101_2"""
        net = wide_resnet101_2(pretrained=True)

    def test_resnet18_in_channels_1(self):
        """Test resnet-18 with in_channels=1"""
        net = resnet18(in_channels=1, pretrained=True)

    def test_resnet18_activation(self):
        """Test resnet-18 with different activation function"""
        activation = {
            'name': 'PReLU',
            'params': {}
        }
        net = resnet18(in_channels=1, pretrained=True, activation=activation)
        for name, module in net.named_modules():                                         
            if name == 'activation':
                assert isinstance(module, nn.PReLU)

            if name == 'layer1.0':
                assert isinstance(module.activation, nn.PReLU)


if __name__ == "__main__":
    unittest.main()
