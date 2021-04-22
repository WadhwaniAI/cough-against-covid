"""Defines EfficientNet backbone for feature extraction."""
from typing import Callable
import torch
from timm.models import efficientnet


class BaseTimmModel(torch.nn.Module):
    """Base class for loading models from timm
    :param method: method to load the architecture class
    :param method: torch.nn.Module
    :param variant: specific architecture to use
    :param variant: str
    :param num_classes: number of classes to be predicted
    :param num_classes: int
    :param in_channels: number of input channels, defaults to 3
    :param in_channels: int, optional
    :param return_features: whether to only return features during inference,
        defaults to False
    :param return_features: bool, optional
    """
    def __init__(
            self, method: Callable, variant: str, num_classes: int,
            in_channels: int = 3, return_features: bool = False):
        super(BaseTimmModel, self).__init__()
        self.net = method(
            pretrained=True, num_classes=num_classes, in_chans=in_channels)
        self.return_features = return_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.return_features:
            return self.net.forward_features(input)

        return self.net(input)


class EfficientNet(BaseTimmModel):
    """EfficientNet with all its variants
    :param variant: specific architecture to use (b0-b7)
    :param variant: str
    :param num_classes: number of classes to be predicted
    :param num_classes: int
    :param in_channels: number of input channels, defaults to 3
    :param in_channels: int, optional
    :param return_features: whether to only return features during inference,
        defaults to False
    :param return_features: bool, optional
    """
    def __init__(
            self, variant: str, num_classes: int, in_channels: int = 3,
            return_features: bool = False):
        method = getattr(efficientnet, variant)
        super(EfficientNet, self).__init__(
            method, variant, num_classes, in_channels, return_features)
