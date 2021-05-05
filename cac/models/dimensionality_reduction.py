"""Defines the base classes to be extended by specific types of models."""
import sys
from os.path import join, exists
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from sklearn.metrics import davies_bouldin_score

from cac.config import Config
from cac.data.dataloader import get_dataloader
from cac.decomposition.methods import factory as method_factory
from cac.models.base import Estimator
from cac.utils.logger import color


class DimensionalityReductionModel(Estimator):
    """Class for dimensionality reduction models (like PCA, TSNE)

    Args:
    :param config: Config object
    :type config: Config
    """
    def __init__(self, config):
        super(DimensionalityReductionModel, self).__init__(config)
        self.method_name = self.model_config['method']['name']
        self.method = method_factory.create(
            self.method_name, **self.model_config['method']['params'])

    def load_data(self, mode: str, batch_size: int = -1, debug: bool = False):
        """Loads data for obtaining low-dimensional representations

        :param mode: split type of data to load (train/val/test)
        :type mode: str
        :returns: tuple of signals and labels
        """
        # using batch_size=-1 will load the entire dataset in one batch
        dataloader, _ = get_dataloader(
            self.config.data, mode, batch_size=batch_size, shuffle=False,
            drop_last=False)
        
        if debug:
            dataset = dataloader.dataset
            culprit_idx = -1
            for i in tqdm(range(len(dataset)), desc='Iterating over dataset to find culprit idx'):
                try:
                    x = dataset[i]
                except:
                    culprit_idx = i
                    break
            if culprit_idx > -1:
                import ipdb; ipdb.set_trace()
                x = dataset[culprit_idx]

        if batch_size > -1:
            signals, labels, items = [], [], []
            iterator = iter(dataloader)
            for j in tqdm(range(len(iterator)), desc='Iterating over batches'):
                batch = next(iterator)
                signals.append(batch['signals']),
                labels.extend(batch['labels'])
                items.extend(batch['items'])
            signals = torch.cat(signals)
        else:
            batch = next(iter(dataloader))
            signals, labels, items = batch['signals'], batch['labels'], batch['items']

        return signals, labels, items

    def fit(self, batch_size: int = 32, debug: bool = False, return_predictions: bool = False, mode: str = 'all'):
        """Fits loaded data to the defined dimensionality reduction model

        :param debug: flag to denote whether or not to run a sample debug run
        :type debug: bool
        :param return_predictions: flag to denote whether or not to return data and predictions
        :type return_predictions: bool
        :param mode: split type of data to load (train/val/test)
        :type mode: str, defaults to 'all'

        :returns: data dict which contains the input data, latent representations and labels
        """
        X, Y, items = self.load_data(mode=mode, debug=debug, batch_size=batch_size)

        self._check_input(X)

        logging.info(color('Learning latent representations', 'blue'))

        X = X.numpy()
        Z = self.method.fit_transform(X)

        data = {
            'input': X,
            'latent': Z,
            'labels': Y,
            'items': items
        }

        if return_predictions:
            return data

    def evaluate(self, X, as_tensor: bool = False):
        """Returns the latent representation for input X"""
        output = self.method.transform(X)

        if as_tensor:
            output = torch.Tensor(output)

        return output

    @staticmethod
    def _check_input(_input):
        assert len(_input.shape) == 2 and isinstance(_input, torch.Tensor)

    @staticmethod
    def _check_outputs(Z, Y):
        assert len(labels) == Z.shape[0]
        assert len(Z.shape) == 2 and Z.shape[1] == 2


class DimensionalityReductionModelBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs):
        if not self._instance:
            self._instance = DimensionalityReductionModel(**kwargs)
        return self._instance
