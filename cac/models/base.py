"""Defines the base classes to be extended by specific types of models."""
import warnings
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import wandb

from cac.utils.logger import color
from cac.data.utils import read_dataset_from_config
from cac.data.dataloader import get_dataloader
from cac.callbacks import ModelCheckpoint
from cac.models.utils import get_subsets


class Estimator(ABC):
    """Defines base class serving as a common machine learning estimator"""
    def __init__(self, config):
        super(Estimator, self).__init__()
        self.config = config
        self.data_config = self.config.data
        self.model_config = self.config.model

    @abstractmethod
    def fit(self):
        """Train the model"""
        pass

    @abstractmethod
    def evaluate(self, data_loader):
        """Evaluate the model on given data

        :param data_loader: data_loader made from the evaluation dataset
        :type data_loader: DataLoader
        """
        pass


class Model(Estimator):
    """Base class for neural network based models

    :param config: Config object
    :type config: Config
    """
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.network_config = self.config.network

        # set default tensor based on device
        if torch.cuda.is_available():
            self.device = 'cuda'
            cudnn.benchmark = True
        else:
            self.device = 'cpu'

        # define network
        self._setup_network()

        # define optimizer
        self._setup_optimizers()

        # maintains a count of the epoch
        self.epoch_counter = 0

        # loads the network and optimizer states based on config
        self.load(self.model_config['load'])

        # freeze layers based on config
        self._freeze_layers()

        # define callbacks
        self._setup_callbacks()

    @abstractmethod
    def _setup_network(self):
        """Setup the network which needs to be trained"""
        pass

    @abstractmethod
    def _freeze_layers(self):
        """Freeze layers based on config during training"""
        pass

    @abstractmethod
    def _setup_optimizers(self):
        """Setup optimizers to be used while training"""
        pass

    def _setup_callbacks(self):
        self.checkpoint = ModelCheckpoint(
            self.config.checkpoint_dir, **self.config.model['save'])

    @abstractmethod
    def calculate_instance_loss(
            self, predictions, targets, mode: str,
            as_numpy: bool = False) -> dict:
        """Calculate loss per instance in a batch

        :param predictions: Predictions (Predicted)
        :type predictions: Any
        :param targets: Targets (Ground Truth)
        :type targets: Any
        :param mode: train/val/test
        :type mode: str
        :param as_numpy: flag to decide whether to return losses as np.ndarray
        :type as_numpy: bool

        :return: dict of losses with list of loss values per instance
        """
        pass

    @abstractmethod
    def calculate_batch_loss(self, instance_losses) -> dict:
        """Calculate mean of each loss for the batch

        :param instance_losses: losses per instance in the batch
        :type instance_losses: dict

        :return: dict containing various loss values over the batch
        """
        pass

    def calculate_epoch_loss(self, loss_dict: dict) -> dict:
        """Calculate mean of each loss for the epoch

        :param loss_dict: dictionary containing arrays of various losses in the epoch
        :type loss_dict: dict

        :return: dict containing various aggregated loss values over the epoch
        """
        epoch_losses = dict()

        for key in loss_dict.keys():
            epoch_losses[key] = np.mean(loss_dict[key])

        return epoch_losses

    def _gather_losses(self, loss_dict: defaultdict) -> dict:
        """Gather all values per loss in one tensor

        :param loss_dict: dictionary containing lists of various losses
        :type loss_dict: defaultdict

        :return: dict containing a running list of various losses per batch
        """
        for loss_name, loss_value in loss_dict.items():
            loss_dict[loss_name] = torch.cat(loss_dict[loss_name]).detach().cpu().numpy()

        return loss_dict

    def _accumulate_losses(self, loss_dict: defaultdict, losses: dict) -> dict:
        """Update the accumulated dict with the given losses

        :param loss_dict: dictionary containing lists of various losses
        :type loss_dict: defaultdict
        :param losses: losses to be added
        :type losses: dict

        :return: dict containing a running list of various losses per batch
        """
        for loss_name, loss_value in losses.items():
            loss_dict[loss_name].append(loss_value.reshape(-1))

        return loss_dict

    @abstractmethod
    def _gather_data(self, epoch_data: dict) -> Tuple:
        """Gather inputs, preds, targets & other epoch data in one tensor

        :param epoch_data: dictionary containing lists of various epoch values
        :type epoch_data: dict

        :return: dictionary with different values as one tensor
        """
        pass

    @abstractmethod
    def _aggregate_data(
            self, epoch_data: dict, method: str,
            at: str, classes: List[str] = None) -> Tuple:
        """Aggregate predictions for a single file by a given `method`

        :param epoch_data: dictionary containing lists of various epoch values
        :type epoch_data: dict
        :param method: method to be used for aggregation, eg. median
        :type method: str
        :param at: point of aggregating the predictions, eg. after softmax
        :type at: str
        :param classes: list of classes in the target
        :type classes: List[str], defaults to None

        :return: epoch data dictionary with values aggregated per file
        """
        pass

    @abstractmethod
    def update_network_params(self, losses: Dict):
        """Defines how to update network weights

        :param losses: losses for the current batch
        :type losses: dict
        """
        pass

    @abstractmethod
    def update_optimizer_params(self, values: dict, update_freq: str):
        """Update optimization parameters like learning rate etc.

        :param values: dictionary of losses and metrics when invoked
            after one epoch or dictionary of losses after one batch
        :type values: dict
        :param update_freq: whether the function is being called after a
            batch or an epoch
        :type update_freq: str
        """
        pass

    @abstractmethod
    def log_batch_summary(self, iterator: Any, mode: str, losses: dict):
        """Logs the summary of the batch on the progressbar in command line

        :param iterator: tqdm iterator
        :type iterator: tqdm
        :param mode: train/val or test mode
        :type mode: str
        :param losses: losses for the current batch
        :type losses: dict
        """
        pass

    @abstractmethod
    def log_epoch_summary(self, mode: str, epoch_losses: dict, metrics: dict,
                          epoch_data: dict, learning_rates: List[Any],
                          batch_losses: defaultdict, instance_losses: defaultdict,
                          use_wandb: bool):
        """Logs the summary of the epoch (losses, metrics and visualizations)

        :param mode: train/val or test mode
        :type mode: str
        :param epoch_losses: aggregate losses aggregated for the epoch
        :type epoch_losses: dict
        :param metrics: metrics for the epoch
        :type metrics: dict
        :param epoch_data: dictionary of various values in the epoch
        :type epoch_data: dict
        :param learning_rates: Dynamically accumulated learning rates per batch
            over all epochs
        :type learning_rates: List[Any]
        :param batch_losses: Dynamically accumulated losses per batch
        :type batch_losses: defaultdict
        :param instance_losses: losses per instance in the batch
        :type instance_losses: defaultdict
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        """
        pass

    @abstractmethod
    def get_subset_data(
            self, epoch_data: dict, indices: List[int],
            instance_losses: defaultdict = None) -> Tuple:
        """Get data for the subset specified by the indices

        :param epoch_data: dictionary of various values in the epoch
        :type epoch_data: dict
        :param indices: list of integers specifying the subset to select
        :type indices: List[int]
        :param instance_losses: losses per instance in the batch
        :type instance_losses: defaultdict, defaults to None

        :return: dict of epoch_data at the given indices
        """
        pass

    @abstractmethod
    def get_eval_params(self, epoch_data: dict) -> Tuple:
        """Get evaluation params by optimizing on the given data

        :param epoch_data: dictionary of various values in the epoch
        :type epoch_data: dict

        :return: dict containing evaluation parameters
        """
        pass

    @abstractmethod
    def compute_epoch_metrics(self, predictions: Any, targets: Any) -> dict:
        """Computes metrics for the epoch

        :param targets: Targets (Ground Truth)
        :type targets: Any
        :param predictions: Predictions (Predicted)
        :type predictions: Any

        :return: dictionary of metrics as provided in the config file
        """
        pass

    @abstractmethod
    def save(self, epoch_metric_values: Dict, use_wandb: bool):
        """Saves the model and optimizer states

        :param epoch_metric_values: validation metrics computed for current epoch
        :type epoch_metric_values: Dict
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        """
        pass

    @abstractmethod
    def load(self, load_config: Dict):
        """Loads the network and optimizer states (optionally) from a config.

        :param load_config: config defining parameters related to
            loading the model and optimizer
        :type load_config: Dict
        """
        pass

    @abstractmethod
    def _accumulate_lr(self, learning_rates: List[Any]) -> dict:
        """Accumulate learning rate values

        :param learning_rates: Dynamically accumulated learning rates per batch
            over all epochs
        :type learning_rates: List[Any]
        :return: dict containing a running list of learning rates
        """
        pass

    @abstractmethod
    def process_batch(self, batch: Any, mode: str = None):
        """Returns the predictions and targets for each batch

        :param batch: one batch of data
        :type batch: Any
        :param mode: train/val/test mode
        :type mode: str

        :return: dict containing predictions and targets
        """
        pass

    def process_epoch(
            self, data_loader: DataLoader, mode: str = None,
            training: bool = False, use_wandb: bool = True,
            log_summary: bool = True, overfit_batch: bool = False,
            compute_metrics: bool = True):
        """Basic epoch function (Used for train/val/test epochs)
        Args:
        :param dataloader: torch DataLoader for the epoch
        :type dataloader: DataLoader
        :param mode: train/val/test mode
        :type mode: str, defaults to None
        :param training: specifies where the model should be in training mode;
            if True, network is set to .train(). Else, it is set to .eval()
        :type training: str, defaults to False
        :param use_wandb: whether to log visualizations to wandb
        :type use_wandb: bool, defaults to True
        :param log_summary: whether to log epoch summary
        :type log_summary: bool, defaults to True
        :param overfit_batch: whether this run is for overfitting on a batch
        :type overfit_batch: bool
        :param compute_metrics: whether to compute metrics
        :type compute_metrics: bool
        """
        instance_losses = defaultdict(list)
        batch_losses = defaultdict(list)

        epoch_data = defaultdict(list)
        learning_rates = []

        if training:
            training_mode = color('train', 'magenta')
            self.network.train()
        else:
            training_mode = color('eval', 'magenta')
            self.network.eval()

        logging.info('{}: {}'.format(
            color('Setting network training mode:', 'blue'),
            color(training_mode)
        ))

        iterator = tqdm(data_loader, dynamic_ncols=True)

        for batchID, batch in enumerate(iterator):
            # process one batch to compute and return the inputs, predictions,
            # ground truth and item in the batch
            batch_data = self.process_batch(batch, mode)

            # calculate loss per instance in the batch
            _instance_losses = self.calculate_instance_loss(
                predictions=batch_data['predictions'],
                targets=batch_data['targets'],
                mode=mode)

            # append batch loss to the list of losses for the epoch
            instance_losses = self._accumulate_losses(
                instance_losses, _instance_losses)

            # calculate loss for the batch
            _batch_losses = self.calculate_batch_loss(_instance_losses)

            # append batch loss to the list of losses for the epoch
            batch_losses = self._accumulate_losses(batch_losses, _batch_losses)

            if mode is not None:
                # log batch summary
                self.log_batch_summary(iterator, mode, _batch_losses)

                # update network weights
                if 'train' in mode:
                    self.update_network_params(_batch_losses)

            # accumulate learning rate before scheduler step
            self._accumulate_lr(learning_rates)

            # update optimizer parameters using schedulers that operate per batch
            # like CyclicalLearningRate
            if hasattr(self, 'update_freq') and 'batch' in self.update_freq and mode == 'train':
                self.update_optimizer_params(_batch_losses, 'batch')

            # accumulate the batch data
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor) \
                    and batch_data[key].device.type:
                    batch_data[key] = batch_data[key].detach().cpu()

                epoch_data[key].append(batch_data[key])

            # ignore other batches after the first batch if we are
            # overfitting a batch
            if overfit_batch:
                break

            # if batchID == 100:
            #     break

        logging.info('Gathering data')
        epoch_data = self._gather_data(epoch_data)

        logging.info('Gathering losses')
        aggregate_params = self.model_config['eval']['aggregate'][mode]

        if len(aggregate_params):
            logging.info(color('Aggregating predictions', 'blue'))

            # aggregate predictions for each file before computing metrics
            epoch_data = self._aggregate_data(epoch_data, **aggregate_params)

            logging.info(
                color('Computing loss on aggregated predictions', 'blue'))

            # gather all instance losses
            # no need to gather losses after this since we are re-computing
            # on the entire dataset once
            instance_losses = self.calculate_instance_loss(
                epoch_data['predictions'],
                epoch_data['targets'], mode=mode, as_numpy=True)

            # pass batch_losses as none since it does not mean anything
            # while aggregating
            batch_losses = None

            # compute epoch_losses based on new instance_losses
            epoch_losses = self.calculate_epoch_loss(instance_losses)

        else:
            # gather all instance losses
            instance_losses = self._gather_losses(instance_losses)

            # gather all batch losses
            batch_losses = self._gather_losses(batch_losses)

            # accumulate list of batch losses to epoch loss
            epoch_losses = self.calculate_epoch_loss(batch_losses)

        logging.info('Computing metrics')

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
                    epoch_data, subset_indices, instance_losses)

                all_data[subset_mode] = subset_data

        results = dict()
        results.update(epoch_losses)
        results['batch_losses'] = batch_losses
        results['instance_losses'] = instance_losses

        if compute_metrics:
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
            logging.info('Computing metrics')

            metrics = self.compute_epoch_metrics(
                epoch_data['predictions'], epoch_data['targets'],
                **eval_params)

            if log_summary:
                logging.info('Logging epoch summary')
                # log losses, metrics and visualizations
                self.log_epoch_summary(
                    mode, epoch_losses, metrics, epoch_data,
                    learning_rates, batch_losses, instance_losses,
                    use_wandb)

                for subset_mode, subset_data in all_data.items():
                    # calculate subset metrics
                    subset_metrics = self.compute_epoch_metrics(
                        subset_data['epoch_data']['predictions'],
                        subset_data['epoch_data']['targets'],
                        **eval_params)

                    # log subset values
                    self.log_epoch_summary(
                        subset_mode, subset_data['epoch_losses'],
                        subset_metrics, subset_data['epoch_data'],
                        None, None, subset_data['instance_losses'], use_wandb)

            # update results
            results.update(metrics)

        for key in epoch_data:
            results[key] = epoch_data[key]
        return results

    def fit(
            self, debug: bool = False, overfit_batch: bool = False,
            use_wandb: bool = True):
        """Entry point to training the network

        :param debug: test run with epoch only on the val set without training
        :type debug: bool
        :param overfit_batch: whether this run is for overfitting on a batch
        :type overfit_batch: bool
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        """
        if not debug:
            # if we are overfitting a batch, then turn off shuffling
            # for the train set or else set it to True
            shuffle = not overfit_batch
            train_dataloader, _ = get_dataloader(
                self.data_config, 'train',
                self.model_config['batch_size'],
                num_workers=self.config.num_workers,
                shuffle=shuffle,
                drop_last=False)

        # ignore val operations when overfitting on a batch
        if not overfit_batch:
            val_dataloader, _ = get_dataloader(
                self.data_config, 'val',
                self.model_config['batch_size'],
                num_workers=self.config.num_workers,
                shuffle=False,
                drop_last=False)
        else:
            logging.info(color('Overfitting a single batch', 'blue'))

        # setup subset trackers
        self.subsets_to_track = defaultdict()
        for mode in self.model_config['subset_tracker']:
            # each mode (train/val) can have multiple subsets that we
            # want to track
            self.subsets_to_track[mode] = get_subsets(
                self.model_config['subset_tracker'][mode])

        # track gradients and weights in wandb
        if use_wandb:
            self.network.watch()

        best_metric_values = None

        for epochID in range(self.model_config['epochs']):
            if not debug:
                # train epoch
                train_results = self.process_epoch(
                    train_dataloader, 'train', training=True, use_wandb=use_wandb,
                    overfit_batch=overfit_batch)

            # ignore val operations when overfitting on a batch
            if not overfit_batch:
                # val epoch
                val_results = self.process_epoch(
                    val_dataloader, 'val', training=False, use_wandb=use_wandb)

                # save best model
                self.save(val_results, use_wandb=use_wandb)

                # update optimizer parameters using schedulers that
                # operate per batch like ReduceLROnPlateau
                if hasattr(self, 'update_freq') and 'epoch' in self.update_freq:
                    logging.info('Running scheduler step')
                    self.update_optimizer_params(val_results, 'epoch')

            # increment epoch counter
            self.epoch_counter += 1

    @abstractmethod
    def evaluate(
            self, data_loader: DataLoader, mode: str, use_wandb: bool = True,
            ignore_cache: bool = True):
        """Evaluate the model on given data

        :param data_loader: data_loader made from the evaluation dataset
        :type data_loader: DataLoader
        :param mode: split of the data represented by the dataloader (train/test/val)
        :type mode: str
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool, defaults to True
        :param ignore_cache: whether to ignore cached values
        :type ignore_cache: bool, defaults to True
        """
        pass

    def _update_wandb(self, mode: str, epoch_losses: dict, metrics: dict):
        """Logs values to wandb

        :param mode: train/val or test mode
        :type mode: str
        :param epoch_losses: aggregate losses aggregated for the epoch
        :type epoch_losses: dict
        :param metrics: metrics for the epoch
        :type metrics: dict
        """
        logging.info('Logging to W&B')
        self.wandb_logs = {}

        for loss, value in epoch_losses.items():
            self.wandb_logs['{}/{}'.format(mode, loss)] = value

        for metric, value in metrics.items():
            # only log metrics with scalar values here
            if isinstance(value, (int, float)):
                self.wandb_logs['{}/{}'.format(mode, metric)] = value
