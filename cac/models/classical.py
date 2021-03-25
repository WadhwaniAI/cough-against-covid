"""Defines the classical ML models building on top of Estimator."""
import sys
from os import makedirs
from os.path import join, exists, dirname, basename, splitext
import logging
from collections import defaultdict
from typing import Any, Dict, Tuple, List
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from sklearn.metrics import precision_recall_curve, accuracy_score,\
    recall_score, precision_score, roc_curve
import matplotlib.pyplot as plt

from cac.data.dataloader import get_dataloader
from cac.data.utils import read_dataset_from_config
from cac.models.base import Estimator
from cac.models.utils import tensorize
from cac.utils.logger import color, set_logger
from cac.utils.io import save_pkl, load_pkl
from cac.sklearn import factory as method_factory
from cac.utils.metrics import PrecisionAtRecall, ConfusionMatrix
from cac.utils.viz import fig2im, plot_lr_vs_loss, plot_classification_metric_curve
from cac.utils.wandb import get_audios, get_images, get_indices, get_confusion_matrix


class ClassicalModel(Estimator):
    """Base class for classical machine-learning based models

    Args:
    :param config: Config object
    :type config: Config
    """
    def __init__(self, config):
        super(ClassicalModel, self).__init__(config)

        self.method = self.load_method(self.model_config['method'])
        self.fixed_epoch = 1

    def load_method(self, method_config: Any):
        """Loads classical method based on config

        :param method_config: method config describing method-design choices (eg. SVM)
        :type method_config: Any
        :returns method: classical method for classification
        """
        method_name = method_config['name']
        logging.info(color(f'Loading {method_name} model', 'blue'))
        method = method_factory.create(method_name, **method_config['params'])
        return method

    def _prepare_data(self, data_loader, mode):
        """Prepares data in np.ndarray format to be ingested into classical methods"""
        dataset = next(iter(data_loader))
        X = dataset['signals'].numpy()
        y = dataset['labels'].numpy()
        items = np.array(dataset['items'])

        self._check_data(X, y, mode)

        data = {
            'X': X,
            'y': y,
            'items': items
        }

        return data

    def load_data(self):
        """Loads train and val datasets"""
        train_dataloader, _ = get_dataloader(
            self.data_config, 'train',
            batch_size=-1,
            num_workers=self.config.num_workers,
            shuffle=True,
            drop_last=False)
        train_data = self._prepare_data(train_dataloader, 'train')
        X_train, y_train, items_train = train_data['X'], train_data['y'], train_data['items']

        val_dataloader, _ = get_dataloader(
            self.data_config, 'val',
            batch_size=-1,
            num_workers=self.config.num_workers,
            shuffle=False,
            drop_last=False)
        val_data = self._prepare_data(val_dataloader, 'val')
        X_val, y_val, items_val = val_data['X'], val_data['y'], val_data['items']

        return (X_train, y_train, items_train), (X_val, y_val, items_val)

    def _check_data(self, X, y, mode):
        """Checks data and its labels before classification"""
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert len(X.shape) == 2 and len(y.shape) == 1

        if mode == 'train':
            assert len(np.unique(y)) > 1, 'Training data has labels of only 1 type.'

    def compute_metrics(
            self, predictions: Any, targets: Any,
            threshold: float = None, recall: float = 0.9) -> dict:
        """Computes metrics for the epoch

        :param targets: ground truth
        :type targets: Any
        :param predictions: model predictions
        :type predictions: Any
        :param threshold: confidence threshold to be used for binary classification; if None,
            the optimal threshold is found.
        :type threshold: float, defaults to None
        :param recall: minimum recall to choose the optimal threshold
        :type recall: float, defaults to 0.9

        :return: dictionary of metrics as provided in the config file
        """
        predictions = tensorize(predictions)
        targets = tensorize(targets)

        # convert to prediction probabilites from logits
        predict_proba = F.softmax(predictions, -1)
        targets = targets.cpu()
        predict_proba = predict_proba.detach().cpu()

        if self.model_config['type'] == 'binary':
            predict_proba = predict_proba[:, 1]

            if threshold is None:
                _, _, threshold = PrecisionAtRecall(recall=recall)(
                    targets, predict_proba)

            predictions = torch.ge(predict_proba, threshold).cpu()
            confusion_matrix = ConfusionMatrix(self.model_config['classes'])
            confusion_matrix(targets, predictions)

            tp = confusion_matrix.tp
            fp = confusion_matrix.fp
            tn = confusion_matrix.tn
            fp = confusion_matrix.fp

            metrics = {
                'accuracy': accuracy_score(targets, predictions),
                'confusion_matrix': confusion_matrix.cm,
                'precision': precision_score(targets, predictions),
                'recall': recall_score(targets, predictions, zero_division=1),
                'threshold': float(threshold),
                'ppv': confusion_matrix.ppv,
                'npv': confusion_matrix.npv,
                'specificity': confusion_matrix.specificity,
                'plr': confusion_matrix.plr,
                'nlr': confusion_matrix.nlr,
                'overall_accuracy': confusion_matrix.overall_accuracy
            }

            precisions, recalls, thresholds = precision_recall_curve(
                targets, predict_proba)
            metrics['pr-curve'] = plot_classification_metric_curve(
                recalls, precisions, xlabel='Recall',
                ylabel='Precision')
            plt.close()

            fprs, tprs, _ = roc_curve(targets, predict_proba)
            metrics['roc-curve'] = plot_classification_metric_curve(
                fprs, tprs, xlabel='False Positive Rate',
                ylabel='True Positive Rate')
            plt.close()

            specificities = np.array([1 - fpr for fpr in fprs])
            metrics['ss-curve'] = plot_classification_metric_curve(
                tprs, specificities, xlabel='Sensitivity',
                ylabel='Specificity')
            plt.close()
        else:
            raise NotImplementedError()

        return metrics

    def _update_wandb(self, mode: str, metrics: dict, predictions: Any,
        targets: Any, items: List[str]):
        """Logs values to wandb

        :param mode: train/val or test mode
        :type mode: str
        :param metrics: metrics for the epoch
        :type metrics: dict
        :param targets: ground truth
        :type targets: Any
        :param predictions: model predictions
        :type predictions: Any
        :param items: list of items of all inputs
        :type items: List[str]
        """
        predictions = tensorize(predictions)
        targets = tensorize(targets)

        wandb_logs = {}

        # decide indices to visualize
        indices = get_indices(targets)

        targets = targets.cpu()

        # log data summary
        f, ax = plt.subplots(1)
        ax.hist(targets)
        ax.set_xticks(np.unique(targets))
        ax.grid()
        wandb.log({
            '{}/data_distribution'.format(mode): wandb.Image(fig2im(f))
        }, step=self.fixed_epoch)
        plt.close()

        # log audios
        audios = get_audios(items[indices], F.softmax(predictions[indices], -1), targets[indices])
        wandb_logs['{}/audios'.format(mode)] = audios

        # log metrics
        display_metrics_keys = ['confusion_matrix', 'pr-curve', 'roc-curve', 'ss-curve']
        numeric_metrics_keys = [k for k in metrics if k not in display_metrics_keys]
        for key in numeric_metrics_keys:
            wandb_logs['{}/{}'.format(mode, key)] = metrics[key]

        wandb_logs['{}/confusion_matrix'.format(mode)] = wandb.Image(
            get_confusion_matrix(metrics['confusion_matrix'],
                                 self.model_config['classes']))
        wandb_logs['{}/pr-curve'.format(mode)] = metrics['pr-curve']
        wandb_logs['{}/roc-curve'.format(mode)] = metrics['roc-curve']
        wandb_logs['{}/ss-curve'.format(mode)] = metrics['ss-curve']

        # log to wandb
        wandb.log(wandb_logs, step=self.fixed_epoch)

    def setup_subsets(self, tracker_cfg: Dict):
        """Sets up subsets to be tracked.

        :param tracker_cfg: config definining all subsets to track
        :type tracker_cfg: Dict
        """
        # setup subset trackers
        subsets_to_track = defaultdict()
        for mode in tracker_cfg:
            mode_subsets = dict()

            # each mode (train/val) can have multiple subsets that we
            # want to track
            for subset_config in tracker_cfg[mode]:
                # each subset has its own data config with a corresponding
                # `mode` and we keep a dictionary of subset `mode` and the
                # corresponding IDs
                subset_info = read_dataset_from_config(subset_config)

                # converting to set as comparison becomes faster than a list
                mode_subsets[subset_config['mode']] = set(subset_info['file'])

            subsets_to_track[mode] = mode_subsets

        return subsets_to_track

    def get_subset_data(
            self, predictions: Any, targets: Any,
            indices: List[int]) -> Tuple:
        """Get data for the subset specified by the indices

        :param targets: Targets for the epoch
        :type targets: Any
        :param predictions: Predictions for the epoch
        :type predictions: Any
        :param indices: list of integers specifying the subset to select
        :type indices: List[int]

        :return: tuple of inputs, predictions and targets at the specified indices
        """
        return predictions[indices], targets[indices]

    def log_summary(
            self, X: np.ndarray, y: np.ndarray, mode: str,
            items: List[Any], use_wandb: bool = True):
        """Computes and logs metrics for given dataset (X, y) and its
        subsets to be tracked

        :param X: input feature matrix
        :type X: np.ndarray
        :param y: labels
        :type y: np.ndarray
        :param mode: train/val/test
        :type mode: str
        :param items: Audio items containing info about the original signal
        :type items: List[Any]
        :use_wandb: flag to decide whether to log to W&B
        :type use_wandb: bool, defaults to True
        """
        y_predict_proba = self.method.predict_proba(X)
        y_predict_proba = np.round(y_predict_proba, decimals=4)
        metrics = self.compute_metrics(y_predict_proba, y)

        if use_wandb:
            self._update_wandb(mode, metrics, y_predict_proba, y, items)

        # track subsets if they exist for the current `mode`
        for subset_mode, subset_paths in self.subsets_to_track[mode].items():
            # match each subset with the larger set using
            # the corresponding IDs (paths)
            subset_indices = [
                index for index, item in enumerate(items)
                if item.path in subset_paths]
            subset_items = items[subset_indices]

            # get the subset data
            subset_y_predict_proba, subset_y = self.get_subset_data(
                y_predict_proba, y, subset_indices)

            # calculate the subset losses and metrics
            subset_metrics = self.compute_metrics(subset_y_predict_proba, subset_y)

            if use_wandb:
                self._update_wandb(subset_mode, subset_metrics,
                    subset_y_predict_proba, subset_y, subset_items)

    def fit(self, use_wandb: bool = True):
        """Entry point to fitting the model to data

        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        """
        # logging.info(color('Loading train and val data ...', 'blue'))
        (X_train, y_train, items_train), (X_val, y_val, items_val) = self.load_data()

        logging.info(color('Setting up subsets to track ...', 'blue'))
        self.subsets_to_track = self.setup_subsets(self.model_config['subset_tracker'])

        logging.info(color('Fitting model to training data ...', 'blue'))
        self.method.fit(X_train, y_train)

        logging.info(color('Logging to W&B ...', 'blue'))
        self.log_summary(X_train, y_train, 'train', items_train, use_wandb)
        self.log_summary(X_val, y_val, 'val', items_val, use_wandb)


    def evaluate(self, data_loader):
        """Evaluate the model on given data

        :param data_loader: data_loader made from the evaluation dataset
        :type data_loader: DataLoader

        :returns metrics: dict with all metrics computed on given data
        """
        data = self._prepare_data(data_loader, 'val')
        X, y = data['X'], data['y']

        y_predict_proba = self.method.predict_proba(X)
        y_predict_proba = np.round(y_predict_proba, decimals=4)

        metrics = self.compute_metrics(y_predict_proba, y)

        return metrics

class ClassicalModelBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs):
        return ClassicalModel(**kwargs)
