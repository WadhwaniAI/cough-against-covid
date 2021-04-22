"""Defines the base classes to be extended by specific analysis modules."""
import warnings
import logging
from os.path import exists, join, splitext
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import wandb

from cac.config import Config
from cac.models import factory as model_factory
from cac.utils.logger import set_logger


class BaseAnalyzer(ABC):
    """Defines base class serving as a basic machine learning model analyzer"""
    def __init__(self, config: Config):
        super(BaseAnalyzer, self).__init__()
        self.check_params(config)
        self.base_config = config

    @staticmethod
    def check_params(config):
        """Checks parameters to the class"""
        assert hasattr(config, 'version')
        assert config.version.endswith('.yml')


class ModelAnalyzer(BaseAnalyzer):
    """Base analyzer class for neural network based models

    :param config: config object on which the model was trained
    :type config: Config
    :param checkpoint: model checkpoint to analyze
    :type checkpoint: int
    :param load_best: flag to decide whether to load best saved model
    :type load_best: bool
    :param debug: flag to decide if it is a sample run
    :type debug: bool, defaults to False
    :param load_attributes: flag to decide whether to load attributes
    :type load_attributes: bool, defaults to True
    """
    def __init__(
            self, config: Config, checkpoint: int, load_best: bool,
            debug: bool = False, load_attributes: bool = True, attr_file_suffix: str = ''):
        super(ModelAnalyzer, self).__init__(config)

        self.set_ckpt(checkpoint, load_best)

        self.model_config = self.base_config.model
        self.data_config = self.base_config.data

        # set logging
        set_logger(join(self.base_config.log_dir, 'analysis.log'))

        # loads model
        self.model = self.load_model(self.base_config)

        if load_attributes:
            # loads data attributes other than usual labels
            self.attributes = self.load_attributes(suffix=attr_file_suffix)

    def set_ckpt(self, checkpoint, load_best):
        """Modifies the model config to load required checkpoints

        :param checkpoint: model checkpoint to analyze
        :type checkpoint: int
        :param load_best: flag to decide whether to load best saved model
        :type load_best: bool
        """
        pass

    def load_model(self, config: Config):
        """Loads model in evaluation mode for analysis

        :param config: config for which the model was trained
        :type config: Config
        :returns: trained model
        """
        model = model_factory.create(config.model['name'], **{'config': config})
        model.network.eval()
        return model

    def load_attributes(self):
        """Loads the attributes other than target labels for all datasets as per config"""
        pass

    def compute_embeddings(
            self, decomposition_cfg: Dict, X: Any, as_tensor: bool = False):
        """Compresses features to 2/3D using PCA/TSNE methods.

        :param decomposition_cfg: dict describing the dimensionality reduction method to use
        :type decomposition_cfg: Dict
        :param X: input data matrix
        :type X: Any
        :param as_tensor: flag to decide whether to return a tensor or numpy array
        """
        pass
