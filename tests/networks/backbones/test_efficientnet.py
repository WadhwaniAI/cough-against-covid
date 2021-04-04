"""Tests cac.networks.backbones.efficientnet"""
import torch
import torch.nn as nn
import unittest
from cac.networks.backbones.efficientnet import EfficientNet


# TODO: enet-b2/4/5/7  do not work on CPU machines (Process is KILLED).
# For now, skipping these tests.
class EfficientNetTestCase(unittest.TestCase):
    """Class to check the EfficientNet backbone"""
    def test_efficientnet_b0(self):
        """Test efficientnet_b0"""
        net = EfficientNet('tf_efficientnet_b0', num_classes=2, in_channels=1)
        dummy = torch.ones((128, 1, 96, 64))
        out = net(dummy)
        self.assertTrue(out.shape, (128, 2))

    # def test_efficientnet_b4(self):
    #     """Test efficientnet_b4"""
    #     net = EfficientNet('tf_efficientnet_b4', num_classes=2, in_channels=1)
    #     dummy = torch.ones((128, 1, 96, 64))
    #     import ipdb; ipdb.set_trace()
    #     out = net(dummy)
    #     self.assertTrue(out.shape, (128, 2))

    # def test_efficientnet_b5(self):
    #     """Test efficientnet_b5"""
    #     net = EfficientNet('tf_efficientnet_b5', num_classes=2, in_channels=1)
    #     dummy = torch.ones((128, 1, 96, 64))
    #     out = net(dummy)
    #     self.assertTrue(out.shape, (128, 2))

    # def test_efficientnet_b7(self):
    #     """Test efficientnet_b7"""
    #     net = EfficientNet('tf_efficientnet_b7', num_classes=2, in_channels=1)
    #     dummy = torch.ones((128, 1, 96, 64))
    #     out = net(dummy)
    #     self.assertTrue(out.shape, (128, 2))

    # def test_efficientnet_features(self):
    #     """Test efficientnet extract_features"""
    #     net = EfficientNet(
    #         'tf_efficientnet_b0', num_classes=2, in_channels=1,
    #         return_features=True)
    #     dummy = torch.ones((128, 1, 224, 224))
    #     out = net(dummy)
    #     self.assertTrue(out.shape, (128, 1280, 7, 7))


if __name__ == "__main__":
    unittest.main()