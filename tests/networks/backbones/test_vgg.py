"""Tests cac.models.backbones.resnet"""
import torch
import unittest
from cac.networks.backbones.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn,\
    vgg13_bn, vgg16_bn, vgg19_bn


class VGGTestCase(unittest.TestCase):
    """Class to check the VGG-based backbones"""
    def test_vgg11(self):
        """Test vgg11"""
        net = vgg11(pretrained=True)

    def test_vgg13(self):
        """Test vgg13"""
        net = vgg13(pretrained=True)

    def test_vgg16(self):
        """Test vgg16"""
        net = vgg16(pretrained=True)

    def test_vgg19(self):
        """Test vgg19"""
        net = vgg19(pretrained=True)

    def test_vgg11_bn(self):
        """Test vgg11_bn"""
        net = vgg11_bn(pretrained=True)

    def test_vgg13_bn(self):
        """Test vgg13_bn"""
        net = vgg13_bn(pretrained=True)

    def test_vgg16_bn(self):
        """Test vgg16_bn"""
        net = vgg16_bn(pretrained=True)

    def test_vgg19_bn(self):
        """Test vgg19_bn"""
        net = vgg19_bn(pretrained=True)

    def test_vgg16_in_channels_1(self):
        """Test vgg16 with in_channels=1"""
        net = vgg16(in_channels=1, pretrained=True)


if __name__ == "__main__":
    unittest.main()
