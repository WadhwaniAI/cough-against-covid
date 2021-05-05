"""Defines base class for defining networks."""
from typing import Any, Dict
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """Base network class which is to be extended by all network architectures."""
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the network

        :param x: input to the network
        :type x: torch.Tensor
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict):
        """Defines helper function to load saved model checkpoint"""
        pass

    @abstractmethod
    def get_state_dict(self) -> OrderedDict:
        """Defines helper function to save saved model checkpoint"""
        pass
