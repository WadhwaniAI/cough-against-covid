"""Defines Factory object to register various layers"""
from typing import Any, List
import math
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import Conv2d, Linear, BatchNorm2d, ReLU,\
    LeakyReLU, MaxPool2d, AdaptiveAvgPool2d, Flatten, Dropout,\
    Sigmoid, Conv1d, BatchNorm1d, MaxPool1d, AdaptiveAvgPool1d, \
    GroupNorm, PReLU, Module, Softmax
from cac.factory import Factory
from cac.networks.backbones.resnet import resnet18, resnet34, resnet50, \
    resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,\
    wide_resnet50_2, wide_resnet101_2
from cac.networks.backbones.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn,\
    vgg13_bn, vgg16_bn, vgg19_bn
from cac.networks.backbones.torchvggish.vggish import VGGish
from siren import Sine


class Swish(Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LinearWithBinaryLabels(Module):
    """Linear layer that operates on both y and ŷ where y is the
    binary label and ŷ are the logits

    Reference: linear adversary for UCI dataset as defined
    here: https://arxiv.org/pdf/1801.07593.pdf

    inputs: y and ŷ
    output (z) = w * [s, sy, s (1-y)] + bias
    where s = sigmoid((1 + |c|) * ŷ)

    Examples::
        >>> m = LinearWithBinaryLabels(20)
        >>> input = torch.randn(128, 2)
        >>> label = torch.randint(0, 2, (128,))
        >>> output = m(input, label)
        >>> print(output.size())
        torch.Size([128, 20])

    :param out_features: size of each output sample
    :param out_features: int
    :param bias: if set to ``False``, the layer will not learn an additive \
        bias; defaults to ``True``
    :param bias: bool
    """
    def __init__(self, out_features: int, bias: bool = True) -> None:
        super(LinearWithBinaryLabels, self).__init__()
        self.out_features = out_features
        self.c = Parameter(torch.Tensor(1))
        self.weight = Parameter(torch.Tensor(out_features, 3))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x, labels = self._check_input(inputs)
        s = torch.sigmoid((1 + torch.abs(self.c)) * x)

        return F.linear(
            torch.stack([s, s * labels, s * (1 - labels)], -1),
            self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'out_features={}, bias={}'.format(
            self.out_features, self.bias is not None)

    @staticmethod
    def _check_input(inputs):
        assert isinstance(inputs, list)

        x, labels = inputs
        ndim = len(x.shape)

        if ndim == 1:
            return x, labels

        if ndim != 2 or x.shape[1] != 2:
            raise ValueError(
                'Invalid input shape: need binary logits as input')

        return x[:, 1], labels


factory = Factory()
factory.register_builder('Conv2d', Conv2d)
factory.register_builder('Linear', Linear)
factory.register_builder('BatchNorm2d', BatchNorm2d)
factory.register_builder('ReLU', ReLU)
factory.register_builder('PReLU', PReLU)
factory.register_builder('LeakyReLU', LeakyReLU)
factory.register_builder('MaxPool2d', MaxPool2d)
factory.register_builder('AdaptiveAvgPool2d', AdaptiveAvgPool2d)
factory.register_builder('Flatten', Flatten)
factory.register_builder('Dropout', Dropout)
factory.register_builder('Sigmoid', Sigmoid)
factory.register_builder('Softmax', Softmax)
factory.register_builder('Conv1d', Conv1d)
factory.register_builder('BatchNorm1d', BatchNorm1d)
factory.register_builder('MaxPool1d', MaxPool1d)
factory.register_builder('AdaptiveAvgPool1d', AdaptiveAvgPool1d)
factory.register_builder('GroupNorm', GroupNorm)
factory.register_builder('resnet18', resnet18)
factory.register_builder('resnet34', resnet34)
factory.register_builder('resnet50', resnet50)
factory.register_builder('resnet101', resnet101)
factory.register_builder('resnet152', resnet152)
factory.register_builder('resnext50_32x4d', resnext50_32x4d)
factory.register_builder('resnext101_32x8d', resnext101_32x8d)
factory.register_builder('wide_resnet50_2', wide_resnet50_2)
factory.register_builder('wide_resnet101_2', wide_resnet101_2)
factory.register_builder('vgg11', vgg11)
factory.register_builder('vgg13', vgg13)
factory.register_builder('vgg16', vgg16)
factory.register_builder('vgg19', vgg19)
factory.register_builder('vgg11_bn', vgg11_bn)
factory.register_builder('vgg13_bn', vgg13_bn)
factory.register_builder('vgg16_bn', vgg16_bn)
factory.register_builder('vgg19_bn', vgg19_bn)
factory.register_builder('Swish', Swish)
factory.register_builder('Sine', Sine)
factory.register_builder('LinearWithBinaryLabels', LinearWithBinaryLabels)
factory.register_builder('vggish', VGGish)
