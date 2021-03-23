"""Defines Factory object to register various activation modules"""
from typing import Any
import torch.nn.functional as F
from torch.nn import ReLU, LeakyReLU, Sigmoid, PReLU, Module
from cac.factory import Factory

from siren import Sine


class Swish(Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


factory = Factory()
factory.register_builder('ReLU', ReLU)
factory.register_builder('PReLU', PReLU)
factory.register_builder('LeakyReLU', LeakyReLU)
factory.register_builder('Sigmoid', Sigmoid)
factory.register_builder('Swish', Swish)
factory.register_builder('Sine', Sine)
