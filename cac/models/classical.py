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
    recall_score, precision_score, roc_curve, auc
import matplotlib.pyplot as plt

from cac.data.dataloader import get_dataloader
from cac.utils.metrics import factory as metric_factory
from cac.data.utils import read_dataset_from_config
from cac.models.base import Estimator
from cac.models.utils import tensorize, get_subsets
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

    def _prepare_data(self, data_loader, mode, debug, batch_size):
        """Prepares data in np.ndarray format to be ingested into classical methods"""
        logging.info(color(f'Loading {mode} data ...', 'blue'))
        if debug:
            dataset = data_loader.dataset
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
            iterator = iter(data_loader)
            for j in tqdm(range(len(iterator)), desc='Iterating over batches'):
                batch = next(iterator)
                signals.append(batch['signals']),
                labels.append(batch['labels'])
                items.extend(batch['items'])
            X = torch.cat(signals).numpy()
            y = torch.cat(labels).numpy()
            items = np.array(items)

        else:
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

    def load_data(self, batch_size: int = -1, debug: bool = False):
        """Loads train and val datasets"""

        train_dataloader, _ = get_dataloader(
            self.data_config, 'train',
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            drop_last=False)
        train_data = self._prepare_data(train_dataloader, 'train', debug, batch_size)
        X_train, y_train, items_train = train_data['X'], train_data['y'], train_data['items']

        val_dataloader, _ = get_dataloader(
            self.data_config, 'val',
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            drop_last=False)
        val_data = self._prepare_data(val_dataloader, 'val', debug, batch_size)
        X_val, y_val, items_val = val_data['X'], val_data['y'], val_data['items']

        return (X_train, y_train, items_train), (X_val, y_val, items_val)

    def _check_data(self, X, y, mode):
        """Checks data and its labels before classification"""
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert len(X.shape) == 2 and len(y.shape) == 1

        if mode == 'train':
            assert len(np.unique(y)) > 1, 'Training data has labels of only 1 type.'

    def get_eval_params(self, epoch_data: dict) -> Tuple:
        """Get evaluation params by optimizing on the given data
        :param epoch_data: dictionary of various values in the epoch
        :type epoch_data: dict
        :return: dict containing evaluation parameters
        """
        metrics = self.compute_metrics(
                epoch_data['predictions'], epoch_data['targets'])
        param_keys = ['recall', 'threshold']
        params = {key: metrics[key] for key in param_keys}
        return params

    def compute_predicted_labels(self, predictions: Any, threshold: float,
            as_logits: bool = False, as_numpy=True, classes: List = None):
        predictions = tensorize(predictions)

        if as_logits:
            # convert to softmax scores from logits
            predictions = F.softmax(predictions, -1)

        predict_proba = predictions.detach().cpu()

        if classes is None:
            classes = self.model_config['classes']

        if len(classes) == 2:
            if len(predict_proba.shape) != 1:
                if len(predict_proba.shape) == 2 and predict_proba.shape[1] == 2:
                    predict_proba = predict_proba[:, 1]

                else:
                    raise ValueError('Acceptable shapes for predict_proba for \
                        binary classification are (N,) and (N, 2). Got \
                        {}'.format(predict_proba.shape))

        predicted_labels = torch.ge(predict_proba, threshold).cpu()

        if as_numpy:
            predicted_labels = predicted_labels.numpy()

        predicted_labels = predicted_labels.astype(int)

        return predicted_labels

    def compute_metrics(
            self, predictions: Any, targets: Any,
            threshold: float = None, recall: float = 0.9,
            as_logits: bool = False, classes: List[str] = None) -> dict:
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
        :param as_logits: whether the predictions are logits; if as_logits=True, the values
            are converted into softmax scores before further processing.
        :type as_logits: bool, defaults to False
        :param classes: list of classes in the target
        :type classes: List[str], defaults to None

        :return: dictionary of metrics as provided in the config file
        """
        print(f"Using threshold={threshold}")
        predictions = tensorize(predictions)
        targets = tensorize(targets)

        if as_logits:
            # convert to softmax scores from logits
            predictions = F.softmax(predictions, -1)

        targets = targets.cpu()
        predict_proba = predictions.detach().cpu()

        if classes is None:
            classes = self.model_config['classes']

        if len(classes) == 2:
            if len(predict_proba.shape) != 1:
                if len(predict_proba.shape) == 2 and predict_proba.shape[1] == 2:
                    predict_proba = predict_proba[:, 1]

                else:
                    raise ValueError('Acceptable shapes for predict_proba for \
                        binary classification are (N,) and (N, 2). Got \
                        {}'.format(predict_proba.shape))

            if threshold is None:
                logging.info('Finding optimal threshold based on: {}'.format(
                    self.model_config['eval']['maximize_metric']))
                maximize_fn = metric_factory.create(
                    self.model_config['eval']['maximize_metric'],
                    **{'recall': recall})
                _, _, threshold = maximize_fn(targets, predict_proba)

            predicted_labels = torch.ge(predict_proba, threshold).cpu()
            confusion_matrix = ConfusionMatrix(classes)
            confusion_matrix(targets, predicted_labels)

            tp = confusion_matrix.tp
            fp = confusion_matrix.fp
            tn = confusion_matrix.tn
            fp = confusion_matrix.fp

            metrics = {
                'accuracy': accuracy_score(targets, predicted_labels),
                'confusion_matrix': confusion_matrix.cm,
                'precision': precision_score(targets, predicted_labels),
                'recall': recall_score(
                    targets, predicted_labels, zero_division=1),
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
            metrics['auc-roc'] = auc(fprs, tprs)
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
            predicted_labels = torch.argmax(predict_proba, -1).cpu()
            confusion_matrix = ConfusionMatrix(classes)
            confusion_matrix(targets, predicted_labels)

            metrics = {
                'accuracy': accuracy_score(targets, predicted_labels),
                'confusion_matrix': confusion_matrix.cm,
            }

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

    def get_subset_data(self, epoch_data: dict, indices: List[int]) -> Tuple:
        """Get data for the subset specified by the indices

        :param targets: Targets for the epoch
        :type targets: Any
        :param predictions: Predictions for the epoch
        :type predictions: Any
        :param indices: list of integers specifying the subset to select
        :type indices: List[int]
        :param epoch_data: dictionary of various values in the epoch
        :type epoch_data: dict

        :return: dict of epoch_data at the given indices
        """
        subset_data = dict()
        _epoch_data = dict()
        for key in epoch_data:
            _epoch_data[key] = epoch_data[key][indices]
        subset_data['epoch_data'] = _epoch_data

        return subset_data

    def log_summary(
            self, X: np.ndarray, y: np.ndarray, mode: str,
            items: List[Any], use_wandb: bool = True, save_cache: bool = True, return_predictions=False):
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
        :param use_wandb: flag to decide whether to log to W&B
        :type use_wandb: bool, defaults to True
        :param save_cache: flag to decide whether to save computed metrics and data
        :type save_cache: bool
        """
        results = {}

        y_predict_proba = self.method.predict_proba(X)
        y_predict_proba = np.round(y_predict_proba, decimals=4)

        epoch_data = {'predictions': y_predict_proba, 'targets': y, 'items': items}
        all_data = {
            mode: {
                'epoch_data': epoch_data
            }
        }

        if hasattr(self, 'subsets_to_track'):
            # track subsets if they exist for the current `mode`
            for subset_mode, subset_paths in self.subsets_to_track[mode].items():
                # match each subset with the larger set using
                # the corresponding IDs (paths)
                subset_indices = [
                    index for index, item in enumerate(epoch_data['items'])
                    if item.path in subset_paths]

                # get the subset data
                subset_data = self.get_subset_data(
                    epoch_data, subset_indices)
                subset_data['subset_indices'] = subset_indices
                all_data[subset_mode] = subset_data

        maximize_mode = self.model_config['eval'].get('maximize_mode', mode)
        maximize_mode = maximize_mode if maximize_mode in all_data else mode
        logging.info(
            'Finding optimal evaluation params based on: {}'.format(
                maximize_mode))
        eval_params = self.get_eval_params(
            all_data[maximize_mode]['epoch_data'])

        # remove mode from all_data
        all_data.pop(mode, None)

        # calculate metrics for the epoch
        logging.info(f'Computing metrics for {mode}')

        metrics = self.compute_metrics(
            epoch_data['predictions'], epoch_data['targets'],
            **eval_params)

        results[mode] = metrics
        results[mode]['predictions'] = epoch_data['predictions']
        results[mode]['predicted_labels'] = self.compute_predicted_labels(epoch_data['predictions'], threshold=eval_params['threshold'])
        results[mode]['targets'] = epoch_data['targets']
        results[mode]['eval_params'] = eval_params

        if use_wandb:
            self._update_wandb(mode, metrics, y_predict_proba, y, items)

        for subset_mode, subset_data in all_data.items():
            # calculate subset metrics
            subset_metrics = self.compute_metrics(
                subset_data['epoch_data']['predictions'],
                subset_data['epoch_data']['targets'],
                **eval_params)

            results[subset_mode] = subset_metrics
            results[subset_mode]['predictions'] = subset_data['epoch_data']['predictions']
            results[subset_mode]['predicted_labels'] = self.compute_predicted_labels(
                subset_data['epoch_data']['predictions'], threshold=eval_params['threshold']
            )
            results[subset_mode]['targets'] = subset_data['epoch_data']['targets']
            results[subset_mode]['eval_params'] = eval_params

            if use_wandb:
                self._update_wandb(subset_mode, subset_metrics,
                    subset_data['epoch_data']['predictions'],
                    subset_data['epoch_data']['targets'],
                    subset_data['epoch_data']['items'])

        if save_cache:
            for _mode in results.keys():
                save_path = join(self.config.log_dir, f'epochwise/{_mode}/{self.fixed_epoch}.pt')
                print(f"- Saving logs for {_mode} at {save_path}")
                save_dict = results[_mode]

                save_dict['paths'] = [item.path for item in items]
                save_dict['start'] = [item.start if hasattr(item, 'start') else None for item in items]
                save_dict['end'] = [item.end if hasattr(item, 'end') else None for item in items]

                makedirs(dirname(save_path), exist_ok=True)
                torch.save(save_dict, save_path)

        return results

    def fit(self, data=None, use_wandb: bool = True, debug: bool = False,
            overfit_batch: bool = False, return_predictions: bool = False
        ):
        """Entry point to fitting the model to data

        :param data: tuple of train and val data, precisely  of the form
            `(X_train, y_train, items_train), (X_val, y_val, items_val)`.
            By default, train and val data will be loaded based on config
        :type data: tuple(tuple), defaults to None
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        """
        if data is not None:
            (X_train, y_train, items_train), (X_val, y_val, items_val) = data
        else:
            (X_train, y_train, items_train), (X_val, y_val, items_val) = self.load_data(
                batch_size=self.model_config.get('batch_size', -1), debug=debug
            )

        logging.info(color('Setting up subsets to track ...', 'blue'))
        # setup subset trackers
        self.subsets_to_track = defaultdict()
        for mode in self.model_config['subset_tracker']:
            # each mode (train/val) can have multiple subsets that we
            # want to track
            self.subsets_to_track[mode] = get_subsets(
                self.model_config['subset_tracker'][mode])

        logging.info(color('Fitting model to training data ...', 'blue'))
        self.method.fit(X_train, y_train)

        logging.info(color('Logging to W&B ...', 'blue'))

        train_results = self.log_summary(X_train, y_train, 'train', items_train, use_wandb)
        val_results = self.log_summary(X_val, y_val, 'val', items_val, use_wandb)

        if return_predictions:
            return train_results, val_results

    def evaluate(self, data, return_predictions=True):
        """Evaluate the model on given data
        """
        (X, y, items) = data

        y_predict_proba = self.method.predict_proba(X)
        y_predict_proba = np.round(y_predict_proba, decimals=4)

        if return_predictions:
            return y_predict_proba

        # metrics = self.compute_metrics(y_predict_proba, y)
        metrics = self.compute_metrics(y_predict_proba, y)

        return metrics


class ClassicalModelBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs):
        return ClassicalModel(**kwargs)
