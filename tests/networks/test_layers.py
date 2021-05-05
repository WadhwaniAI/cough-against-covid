"""Tests cac.networks.layers.Layer"""
import unittest
import torch
import torch.nn.functional as F
from numpy.testing import assert_array_equal
from cac.networks.layers import factory as layer_factory


class LayerTestCase(unittest.TestCase):
    """Class to check the creation of Layer"""
    def test_linear(self):
        """Test creation of a linear layer"""
        layer_name = 'Linear'
        args = {
            'in_features': 16,
            'out_features': 8
        }

        layer = layer_factory.create(layer_name, **args)
        self.assertEqual(layer._get_name(), layer_name)

        x = torch.zeros(args['in_features'])
        y = layer(x)
        self.assertEqual(y.shape, torch.Size([args['out_features']]))

    def test_swish(self):
        """Test swish activation function"""
        layer_name = 'Swish'
        args = {}

        layer = layer_factory.create(layer_name, **args)
        x = torch.rand(10)
        out = layer(x)
        target = x * torch.sigmoid(x)

        assert_array_equal(target, out)

    def test_linear_with_binary_labels_2d(self):
        """Tests LinearWithBinaryLabels layer with 2D input"""
        layer_name = 'LinearWithBinaryLabels'
        args = {'out_features': 10}

        layer = layer_factory.create(layer_name, **args)
        x = torch.randn(128, 2)
        y = torch.randint(0, 2, (128,))
        out = layer([x, y])

        self.assertTrue(out.shape, (128, 10))

    def test_linear_with_binary_labels_1d(self):
        """Tests LinearWithBinaryLabels layer with 1D input"""
        layer_name = 'LinearWithBinaryLabels'
        args = {'out_features': 10}

        layer = layer_factory.create(layer_name, **args)
        x = torch.randn(128)
        y = torch.randint(0, 2, (128,))
        out = layer([x, y])

        self.assertTrue(out.shape, (128, 10))

    def test_linear_with_binary_labels_nd(self):
        """Tests LinearWithBinaryLabels layer with 2D input non-binary"""
        layer_name = 'LinearWithBinaryLabels'
        args = {'out_features': 10}

        layer = layer_factory.create(layer_name, **args)
        x = torch.randn(128, 10)
        y = torch.randint(0, 2, (128,))

        with self.assertRaises(ValueError):
            out = layer([x, y])

    def test_conv2d(self):
        """Test creation of a Conv2D layer"""
        layer_name = 'Conv2d'
        args = {
            'in_channels': 1,
            'out_channels': 16,
            'kernel_size': [3, 7]
        }

        layer = layer_factory.create(layer_name, **args)
        self.assertEqual(layer._get_name(), layer_name)

        x = torch.zeros((1, args['in_channels'], 128, 1024))
        y = layer(x)
        self.assertEqual(y.shape, torch.Size([1, args['out_channels'], 126, 1018]))

    def test_batchnorm2d(self):
        """Test creation of a batchnorm2d layer"""
        layer_name = 'BatchNorm2d'
        args = {
            'num_features': 3,
        }

        layer = layer_factory.create(layer_name, **args)
        self.assertEqual(layer._get_name(), layer_name)

        x = torch.zeros((1, args['num_features'], 128, 1024))
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_dropout(self):
        """Test creation of a dropout layer"""
        layer_name = 'Dropout'
        args = {
            'p': 0.5,
        }

        layer = layer_factory.create(layer_name, **args)
        self.assertEqual(layer._get_name(), layer_name)

        x = torch.zeros((1, 1, 128, 1024))
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_maxpool2d(self):
        """Test creation of a maxpool2d layer"""
        layer_name = 'MaxPool2d'
        args = {
            'kernel_size': [3, 7],
        }

        layer = layer_factory.create(layer_name, **args)
        self.assertEqual(layer._get_name(), layer_name)

        x = torch.zeros((1, 16, 128, 1024))
        y = layer(x)
        self.assertEqual(y.shape, torch.Size([1, 16, 42, 146]))

    def test_adaptiveavgpool2d(self):
        """Test creation of a adaptiveavgpool2d layer"""
        layer_name = 'AdaptiveAvgPool2d'
        args = {
            'output_size': [1, 1],
        }

        layer = layer_factory.create(layer_name, **args)
        self.assertEqual(layer._get_name(), layer_name)

        x = torch.zeros((1, 16, 128, 1024))
        y = layer(x)
        self.assertEqual(y.shape, torch.Size([1, 16, *args['output_size']]))

    def test_flatten(self):
        """Test creation of a Flatten layer"""
        layer_name = 'Flatten'
        args = {}

        layer = layer_factory.create(layer_name, **args)
        self.assertEqual(layer._get_name(), layer_name)

        x = torch.zeros((1, 16, 16))
        y = layer(x)
        self.assertEqual(y.shape, torch.Size([1, 256]))


if __name__ == "__main__":
    unittest.main()
