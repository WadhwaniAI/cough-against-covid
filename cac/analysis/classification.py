"""Defines the base classes to be extended by specific types of models."""
import sys
from os import makedirs
from os.path import join, exists, dirname, splitext, basename
import logging
from glob import glob
import multiprocessing as mp
from collections import defaultdict
from typing import Any, Dict, Tuple, List
from tqdm import tqdm
import numpy as np
import pandas as pd
from natsort import natsorted
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from cac.config import Config, DATA_ROOT
from cac.analysis.utils import get_audio_type, get_unique_id
from cac.utils.io import read_yml, load_pkl
from cac.utils.logger import color
from cac.decomposition.methods import factory as decomposition_factory
from cac.analysis.base import ModelAnalyzer


class ClassificationAnalyzer(ModelAnalyzer):
    """Base analyzer class for neural network based models

    :param config: config object on which the model was trained
    :type config: Config
    :param checkpoint: model checkpoint to analyze
    :type checkpoint: int
    :param load_best: flag to decide whether or not to load best model
    :type load_best: bool
    :param debug: flag to decide if it is a sample run, only loads val dataloader
    :type debug: bool
    :param load_attributes: flag to decide whether to load attributes
    :type load_attributes: bool, defaults to True

    # TOOD: Loading attributes is specific to wiai-* datasets.
    # TODO: FIX-`pandas.DataFrame.combine_first` causes `NumExpr defaulting to 4 threads.`
    """
    def __init__(self, config: Config, checkpoint: int, load_best: bool, debug: bool = False, load_attributes: bool = True):
        super(ClassificationAnalyzer, self).__init__(
            config, checkpoint, load_best, debug, load_attributes)

    def set_ckpt(self, checkpoint, load_best):
        """Modifies the model config to load required checkpoints

        :param checkpoint: model checkpoint to analyze
        :type checkpoint: int
        :load_best: flag to decide whether to load best saved model
        :type load_best: bool
        """
        self.base_config.model['load'] = {
            'epoch': checkpoint,
            'load_best': load_best,
            'resume_epoch': False,
            'resume_optimizer': False,
            'version': splitext(self.base_config.version)[0]
        }

    def load_epochwise_logs(self, mode: str, ext='pt', get_metrics: bool = True):
        """Load instance and batch level logs for all epochs.

        :param mode: train/val/test
        :type mode: str
        """
        log_dir = join(self.base_config.log_dir, 'epochwise', mode)
        all_logfiles = natsorted(glob(join(log_dir, f'*.{ext}')))

        instance_losses = defaultdict(list)
        predict_probs = defaultdict(list)
        predict_labels = defaultdict(list)
        batch_losses = defaultdict(list)
        thresholds = defaultdict(list)
        numeric_metrics = defaultdict(list)
        display_metrics = defaultdict(list)

        for file in tqdm(all_logfiles, dynamic_ncols=True, desc='Loading logs'):
            epochID = splitext(basename(file))[0]
            key = 'epoch_{}'.format(epochID)

            if ext == 'pkl':
                epoch_logs = load_pkl(file)
            elif ext == 'pt':
                epoch_logs = torch.load(file)

            paths = list(epoch_logs['paths'])
            audio_types = [get_audio_type(path) for path in paths]
            unique_ids = [get_unique_id(path) for path in paths]
            targets = epoch_logs['targets'].tolist()

            for df in [instance_losses, predict_probs, predict_labels, batch_losses]:
                df['audio_type'] = audio_types
                df['unique_id'] = unique_ids
                df['targets'] = targets

            instance_losses[key] = epoch_logs['instance_loss'].tolist()
            _predict_probs = F.softmax(epoch_logs['predictions'], -1)
            predict_probs[key] = _predict_probs.tolist()

            if get_metrics:
                metrics = self.compute_metrics(
                    _predict_probs, np.array(targets)
                )
                numeric_metrics[key], display_metrics[key] = metrics

            if 'batch_loss' in epoch_logs:
                batch_losses[key] = epoch_logs['batch_loss'].tolist()

            if 'threshold' in epoch_logs:
                if self.model_config['type'] == 'binary':
                    predict_labels[key] = _predict_probs[:, 1].ge(
                        epoch_logs['threshold']).int().tolist()

                thresholds[key] = epoch_logs['threshold']

        attributes = self.match_attributes_by_paths(paths)

        results = dict()
        results['instance_loss'] = pd.DataFrame(instance_losses)
        results['predict_probs'] = pd.DataFrame(predict_probs)
        results['predict_labels'] = pd.DataFrame(predict_labels)
        results['batch_loss'] = pd.DataFrame(batch_losses)
        results['attributes'] = pd.DataFrame(attributes)
        results['thresholds'] = thresholds
        results['targets'] = targets
        results['numeric_metrics'] = numeric_metrics
        results['display_metrics'] = display_metrics

        return results

    def compute_metrics(
            self, predictions: Any, targets: Any,
            threshold: float = None, recall: float = 0.9,
            as_logits: bool = False) -> dict:
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

        :return: dictionary of metrics as provided in the config file
        """
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.Tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.Tensor(targets)

        all_metrics = self.model.compute_epoch_metrics(
            predictions, targets, threshold=threshold,
            recall=recall, as_logits=as_logits)

        display_metrics_keys = ['confusion_matrix', 'pr-curve', 'roc-curve', 'ss-curve']
        display_metrics = dict((k, all_metrics[k]) for k in tuple(display_metrics_keys))

        numeric_metrics_keys = [k for k in all_metrics if k not in display_metrics_keys]
        numeric_metrics = dict((k, all_metrics[k]) for k in tuple(numeric_metrics_keys))

        return numeric_metrics, display_metrics

    def load_attributes(self, suffix):
        """Loads the attributes other than target labels for all datasets as per config"""
        dataset_cfg = self.data_config['dataset']['config']

        sheets = []
        num_rows, cols = 0, set()
        for dataset in dataset_cfg:
            sheet = pd.read_csv(
                join(DATA_ROOT, dataset['name'], f'processed/attributes{suffix}.csv'))
            sheets.append(sheet)

            num_rows += sheet.shape[0]
            cols = cols.union(set(sheet.columns))

        common_sheet = pd.concat(sheets, axis=0)

        if {'user_id', 'patient_id'}.issubset(common_sheet.columns):
            common_sheet['unique_id'] = common_sheet['user_id'].combine_first(
                common_sheet['patient_id'])
        else:
            key = 'user_id' if 'user_id' in common_sheet.columns else 'patient_id'
            common_sheet['unique_id'] = common_sheet[key]

        common_sheet = common_sheet.set_index('unique_id')
        self._check_sheet(common_sheet, num_rows, cols)

        return common_sheet

    @staticmethod
    def _check_sheet(common_sheet, num_rows, cols):
        """Checks proper concatenttion of `sheets` into a `common_sheet`"""
        assert common_sheet.shape[0] == num_rows
        assert common_sheet.shape[1] == len(cols)

    def match_attributes_by_paths(self, paths):
        audio_types = [get_audio_type(path) for path in paths]
        unique_ids = [get_unique_id(path) for path in paths]

        attributes = []
        for audio_type, unique_id in zip(audio_types, unique_ids):
            try:
                row = self.attributes.loc[unique_id]
                file = row['{}_path'.format(audio_type)]
                row_dict = dict(row)
                row_dict.update(
                    {
                        'file': file,
                        'audio_type': audio_type,
                        'unique_id': unique_id,
                    }
                )
                attributes.append(row_dict)
            except:
                import ipdb; ipdb.set_trace()

        return attributes

    def process_batch(self, batch: Any, last_layer_index: int) -> Tuple[Any, Any]:
        """Returns the predictions and targets for each batch

        :param batch: one batch of data containing inputs and targets
        :type batch: Any
        :param last_layer_index: index of the layer out of which features are required
        :param last_layer_index: int, defaults to -1

        :return: dict containing predictions and targets
        """
        inputs = batch['signals'].to(self.model.device)
        labels = batch['labels'].to(self.model.device)
        paths = [item.path for item in batch['items']]
        attributes = self.match_attributes_by_paths(paths)

        with torch.no_grad():
            features = self.model.network.blocks[:last_layer_index](inputs)

        batch_data = {
            'inputs': inputs,
            'features': features,
            'labels': labels,
            'attributes': attributes,
            'paths': paths
        }

        return batch_data

    def compute_features(self, data_loader: DataLoader, use_wandb: bool = False,
                         last_layer_index: int = -1):
        """Basic epoch function to compute model-based features for data

        Args:
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        :param last_layer_index: index of the layer out of which features are required
        :param last_layer_index: int, defaults to -1
        """
        iterator = tqdm(data_loader, dynamic_ncols=True,
                        desc=color('Computing features'))
        instance_losses = defaultdict(list)
        batch_losses = defaultdict(list)

        all_features = []
        all_targets = []
        all_paths = []
        all_inputs = []
        all_attributes = []

        for batchID, batch in enumerate(iterator):
            # process one batch to compute and return the inputs, features,
            # labels and paths of each file in the batch
            batch_data = self.process_batch(batch, last_layer_index)

            all_inputs.append(batch_data['inputs'])
            all_features.append(batch_data['features'])
            all_targets.append(batch_data['labels'])
            all_paths.extend(batch_data['paths'])
            all_attributes.extend(batch_data['attributes'])

        all_targets = torch.cat(all_targets)
        all_features = torch.cat(all_features)
        all_paths = np.array(all_paths)

        results = dict()
        results['paths'] = all_paths
        results['inputs'] = all_inputs
        results['targets'] = all_targets
        results['features'] = all_features
        results['attributes'] = all_attributes

        return results

    def get_attributes(self, data_loader: DataLoader, as_df=True):
        """Iterates batchwise over data loader to extract attributes.

        :param as_df: return as pd.DataFrame
        :type as_df: bool
        """
        iterator = tqdm(
            data_loader, dynamic_ncols=True, desc=color('Extracting attributes')
        )

        all_attributes = []
        for batchID, batch in enumerate(iterator):
            paths = [item.path for item in batch['items']]
            attributes = self.match_attributes_by_paths(paths)
            all_attributes.extend(attributes)

        if as_df:
            all_attributes = pd.DataFrame(all_attributes)

        return all_attributes

    def compute_embeddings(
            self, decomposition_cfg: Dict, X: Any, as_tensor: bool = False):
        """Compresses features to 2/3D using PCA/TSNE methods.

        :param decomposition_cfg: dict describing the dimensionality reduction method to use
        :type decomposition_cfg: Dict
        :param X: input data matrix, can be List, torch.Tensor or numpy.ndarray
        :type X: Any
        :param as_tensor: flag to decide whether to return a tensor or numpy array
        """
        self.decomposition_method = decomposition_factory.create(
            decomposition_cfg['name'],
            **decomposition_cfg['params'])

        X = self._check_compute_embeddings_input(X)
        Z = self.decomposition_method.fit_transform(X)

        if as_tensor:
            Z = torch.Tensor(Z)

        return Z

    @staticmethod
    def _check_compute_embeddings_input(_input):
        if isinstance(_input, list):
            _input = np.array(_input)
        elif isinstance(_input, torch.Tensor):
            _input = _input.cpu().numpy()

        assert len(_input.shape) == 2
        assert isinstance(_input, np.ndarray)

        return _input
