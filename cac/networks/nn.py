"""Defines the class for neural networks."""
from typing import List
from collections import OrderedDict
import logging

import torch
import torch.nn as nn
import wandb

from cac.networks.base import BaseNetwork
from cac.networks.layers import factory as layer_factory
from cac.networks.init import factory as init_factory
from cac.utils.typing import LayerConfigDict
from cac.utils.logger import color
from cac.networks.backbones.utils import _correct_state_dict


class NeuralNetwork(BaseNetwork):
    """This class constructs a neural network given an architecture config

    :param config: config for defining the network
    :type config: List[LayerConfigDict]
    :param init: params for initializing the parameters of the network
    :type init: dict, defaults to None
    """
    def __init__(self, config: List[LayerConfigDict], init: dict = None):
        super(NeuralNetwork, self).__init__()
        self.config = config
        self.init = init
        self._build_modules()
        self._init_modules()

    def _build_modules(self):
        """Defines method to build all sub-modules of the model"""
        self.blocks = nn.Sequential()
        for index, layer_config in enumerate(self.config):
            layer = layer_factory.create(
                layer_config['name'], **layer_config['params'])
            self.blocks.add_module(
                name='{}_{}'.format(index, layer_config['name'].lower()),
                module=layer)

    def _init_modules(self):
        """Initializes the parameters based on config"""
        if self.init is None:
            return

        logging.info(color('Initializing the parameters'))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_param(m.weight, 'weight')
                if m.bias is not None:
                    self._init_param(m.bias, 'bias')

            elif isinstance(m, nn.BatchNorm2d):
                self._init_param(m.weight, 'bn_weight')
                self._init_param(m.bias, 'bn_bias')

    def freeze_layers(self):
        # freeze layers based on config
        for name, param in self.blocks.named_parameters():
            layer_index = int(name.split('_')[0])
            if not self.config[layer_index].get('requires_grad', True):
                logging.info('Freezing layer: {}'.format(name))
                param.requires_grad = False

    def _init_param(self, tensor, key):
        tensor_init_config = self.init.get(key)
        if 'name' in tensor_init_config:
            tensor_init_params = tensor_init_config.get('params', {})
            tensor_init_params['tensor'] = tensor
            init_factory.create(
                tensor_init_config['name'], **tensor_init_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the network

        :param x: input to the network
        :type x: torch.Tensor

        :return: output on forward pass of input x
        """
        return self.blocks(x)

    def load_state_dict(self, state_dict: OrderedDict):
        """Defines helper function to load saved model checkpoint"""
        try:
            self.blocks.load_state_dict(state_dict)
        except RuntimeError:
            logging.info(color(
                'state_dict does not match strictly. Trying to correct',
                'red'))
            state_dict = _correct_state_dict(
                state_dict, self.blocks.state_dict())
            self.blocks.load_state_dict(state_dict, strict=False)

    def get_state_dict(self) -> OrderedDict:
        """Defines helper function to save saved model checkpoint"""
        return self.blocks.state_dict()

    def watch(self):
        """Defines how to track gradients and weights in wandb"""
        wandb.watch(self.blocks, log='all')


class NeuralNetworkBuilder:
    """Builds a NeuralNetwork object"""
    def __call__(self, **kwargs: dict) -> NeuralNetwork:
        """Builds a NeuralNetwork object

        :param **kwargs: dictionary containing values corresponding to the arguments of
            the NeuralNetwork class
        :type **kwargs: dict
        :returns: a NeuralNetwork object
        """
        return NeuralNetwork(**kwargs)
