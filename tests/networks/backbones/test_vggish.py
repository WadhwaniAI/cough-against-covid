"""Tests cac.models.backbones.torchvggish"""
import torch
import torch.nn as nn
import unittest
from cac.networks.backbones.torchvggish.vggish import VGGish


class VGGishTestCase(unittest.TestCase):
    """Class to check the VGGish backbone"""
    def test_vggish(self):
        """Test VGGish"""
        net = VGGish(pretrained=True, preprocess=False, postprocess=True)
        dummy = torch.ones((128, 1, 96, 64))
        out = net(dummy)
        self.assertTrue(out.shape, (128, 128))


if __name__ == "__main__":
    unittest.main()
