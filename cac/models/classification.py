"""Defines the base classes to be extended by specific types of models."""
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
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from sklearn.metrics import precision_recall_curve, accuracy_score,\
    recall_score, precision_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from cac.data.audio import AudioItem
from cac.data.dataloader import get_dataloader
from cac.models.base import Model
from cac.utils.logger import color
from cac.utils.io import save_pkl, load_pkl
from cac.utils.metrics import ConfusionMatrix
from cac.utils.metrics import factory as metric_factory
from cac.models.utils import get_saved_checkpoint_path, logit
from cac.utils.viz import fig2im, plot_lr_vs_loss, plot_classification_metric_curve
from cac.utils.wandb import get_audios, get_images, get_indices, get_confusion_matrix
from cac.utils.loss import loss_factory
from cac.networks import factory as network_factory
from cac.optimizer import optimizer_factory, scheduler_factory
from evaluation.utils import _save_eval_data
np.set_printoptions(suppress=True)


class ClassificationModel(Model):
    """Classification model class

    Args:
    :param config: Config object
    :type config: Config
    """
    def __init__(self, config):
        super(ClassificationModel, self).__init__(config)
        logging.info(color('Using loss functions:'))
        logging.info(self.model_config.get('loss'))

    def _setup_network(self):
        """Setup the network which needs to be trained"""
        logging.info(color("Building the network"))
        self.network = network_factory.create(
            self.network_config['name'], **self.network_config['params']).to(
            self.device)

    def _freeze_layers(self):
        """Freeze layers based on config during training"""
        logging.info(color('Freezing specified layers'))
        self.network.freeze_layers()

    def _setup_optimizers(self):
        """Setup optimizers to be used while training"""
        if 'optimizer' not in self.model_config:
            return

        logging.info(color("Setting up the optimizer ..."))
        kwargs = self.model_config['optimizer']['args']
        kwargs.update({'params': self.network.parameters()})
        self.optimizer = optimizer_factory.create(
            self.model_config['optimizer']['name'],
            **kwargs)

        if 'scheduler' in self.model_config['optimizer']:
            scheduler_config = self.model_config['optimizer']['scheduler']
            scheduler_config['params']['optimizer'] = self.optimizer
            self.scheduler = scheduler_factory.create(
                scheduler_config['name'],
                **scheduler_config['params'])
            self.update_freq = [scheduler_config['update']]

            if 'value' in scheduler_config:
                self.value_to_track = scheduler_config['value']

    def calculate_instance_loss(
            self, predictions: torch.FloatTensor, targets: torch.LongTensor,
            mode: str, as_numpy: bool = False) -> dict:
        """Calculate loss per instance in a batch

        :param predictions: Predictions (Predicted)
        :type predictions: torch.FloatTensor
        :param targets: Targets (Ground Truth)
        :type targets: torch.LongTensor
        :param mode: train/val/test
        :type mode: str
        :param as_numpy: flag to decide whether to return losses as np.ndarray
        :type as_numpy: bool

        :return: dict of losses with list of loss values per instance
        """
        loss_config = self.model_config.get('loss')[mode]
        criterion = loss_factory.create(
            loss_config['name'], **loss_config['params'])
        loss = criterion(predictions, targets.long())

        if as_numpy:
            loss = loss.cpu().numpy()

        return {'loss': loss}

    def calculate_batch_loss(self, instance_losses) -> dict:
        """Calculate mean of each loss for the batch

        :param batch_losses: losses per instance in the batch
        :type batch_losses: dict

        :return: dict containing various loss values over the batch
        """
        losses = dict()
        for key in instance_losses:
            losses[key] = torch.mean(instance_losses[key])

        return losses

    def _gather_data(self, epoch_data: dict) -> Tuple:
        """Gather inputs, preds, targets & other epoch data in one tensor

        :param epoch_data: dictionary containing lists of various epoch values
        :type epoch_data: dict

        :return: dictionary with different values as one tensor
        """
        epoch_data['inputs'] = torch.cat(epoch_data['inputs']).cpu().numpy()
        epoch_data['predictions'] = torch.cat(
            epoch_data['predictions']).detach().cpu()
        epoch_data['targets'] = torch.cat(epoch_data['targets']).detach().cpu()
        epoch_data['items'] = np.hstack(epoch_data['items'])
        return epoch_data

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

        :return: epoch data dictionary with values aggregated per file
        """
        items = epoch_data['items']
        predictions, targets = epoch_data['predictions'], epoch_data['targets']

        if classes is None:
            classes = self.model_config['classes']

        agg_items, agg_predictions, agg_targets = [], [], []
        
        paths = np.array([item.path for item in items])
        _, idx = np.unique(paths, return_index=True)
        unique_paths = np.take(paths, np.sort(idx))

        if at == 'softmax':
            predictions = F.softmax(predictions, -1)
        elif at == 'sigmoid':
            predictions = torch.sigmoid(predictions.squeeze())
        else:
            raise NotImplementedError

        if len(classes) == 2:
            # only works for binary classification as of now
            if at == 'softmax':
                predictions = predictions[:, 1]
            indices = defaultdict()
            for path in unique_paths:
                indices = np.where(paths == path)[0]

                file_items = items[indices]
                starts = [item.start for item in file_items]
                sorted_indices = indices[np.argsort(starts)]
                file_item = AudioItem(path=path)
                agg_items.append(file_item)

                file_target = targets[sorted_indices][0]
                agg_targets.append(file_target)

                file_predictions = predictions[sorted_indices]
                if method == 'median':
                    file_prediction = np.median(file_predictions)
                elif method == 'mean':
                    file_prediction = torch.mean(file_predictions)
                elif method == 'max':
                    file_prediction = torch.max(file_predictions)
                else:
                    raise NotImplementedError

                if at == 'softmax':
                    inverse_softmax = torch.Tensor([0.0, logit(file_prediction)])
                    agg_predictions.append(inverse_softmax)
                elif at =='sigmoid':
                    inverse_sigmoid = torch.Tensor([logit(file_prediction)])
                    agg_predictions.append(inverse_sigmoid)
        else:
            raise NotImplementedError

        epoch_data['targets'] = torch.stack(agg_targets)
        epoch_data['predictions'] = torch.stack(agg_predictions)
        epoch_data['items'] = np.array(agg_items)

        return epoch_data

    def update_network_params(self, losses):
        """Defines how to update network weights

        Args:
        :param losses: losses for the current batch
        :type losses: dict
        """
        self.optimizer.zero_grad()
        losses['loss'].backward()
        self.optimizer.step()

    def update_optimizer_params(self, values: dict, update_freq: str):
        """Update optimization parameters like learning rate etc.

        :param values: dictionary of losses and metrics when invoked
            after one epoch or dictionary of losses after one batch
        :type values: dict
        :param update_freq: whether the function is being called after a
            batch or an epoch
        :type update_freq: str
        """
        if hasattr(self, 'scheduler'):
            if hasattr(self, 'value_to_track'):
                self.scheduler.step(values[self.value_to_track])
            else:
                self.scheduler.step()

    def log_batch_summary(self, iterator: Any, mode: str, losses: dict):
        """Logs the summary of the batch on the progressbar in command line

        :param iterator: tqdm iterator
        :type iterator: tqdm
        :param mode: train/val or test mode
        :type mode: str
        :param losses: losses for the current batch
        :type losses: dict
        """
        iterator.set_description(
             "V: {} | Epoch: {} | {} | Loss {:.4f}".format(
                self.config.version, self.epoch_counter, mode.capitalize(),
                losses['loss']
                ), refresh=True)

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
        :type instance_losses: dict
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        """
        logging.info(
            color("V: {} | Epoch: {} | {} | Avg. Loss {:.4f}".format(
                self.config.version, self.epoch_counter, mode.capitalize(),
                epoch_losses['loss']
            ), 'green')
        )

        metric_log = "V: {} | Epoch: {} | {}".format(
                    self.config.version, self.epoch_counter, mode.capitalize())

        for metric in self.config.metrics_to_track:
            metric_log += ' | {}: {:.4f}'.format(metric, metrics[metric])

        logging.info(color(metric_log, 'green'))

        # update wandb
        if use_wandb:
            self._update_wandb(
                mode, epoch_losses, metrics, epoch_data, learning_rates,
                batch_losses)

        if batch_losses is not None:
            # reshape batch losses to the shape of instance losses
            instance_batch_losses = dict()
            for loss_name, loss_value in batch_losses.items():
                loss_value = loss_value.reshape(-1, 1)
                loss_value = np.repeat(
                    loss_value, self.model_config['batch_size'],
                    axis=-1).reshape(-1)

                # correct for incomplete last batch
                instance_batch_losses[loss_name] = loss_value[:len(
                    epoch_data['items'])]

        # log instance-level epochwise values
        instance_values = {
            'paths': [item.path for item in epoch_data['items']],
            'predictions': epoch_data['predictions'],
            'targets': epoch_data['targets'],
        }

        for key, value in metrics.items():
            instance_values[key] = value

        starts = [item.start if hasattr(item, 'start') else None for item in epoch_data['items']]
        ends = [item.end if hasattr(item, 'end') else None for item in epoch_data['items']]

        instance_values['start'] = starts
        instance_values['end'] = ends

        for loss_name in instance_losses:
            instance_values['instance_loss'] = instance_losses[loss_name]
            if batch_losses is not None:
                instance_values['batch_loss'] = instance_batch_losses[loss_name]

        save_path = join(self.config.log_dir, 'epochwise', '{}/{}.pt'.format(
            mode, self.epoch_counter))
        makedirs(dirname(save_path), exist_ok=True)
        torch.save(instance_values, save_path)

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
        subset_data = dict()
        _epoch_data = dict()
        for key in epoch_data:
            _epoch_data[key] = epoch_data[key][indices]
        subset_data['epoch_data'] = _epoch_data

        if instance_losses is not None:
            # get the instance losses for the subset
            subset_data['instance_losses'] = dict()
            for loss_name, loss_value in instance_losses.items():
                subset_data['instance_losses'][loss_name] = loss_value[indices]

            # calculate the subset losses and metrics
            subset_data['epoch_losses'] = self.calculate_epoch_loss(
                subset_data['instance_losses'])
        return subset_data

    def get_eval_params(self, epoch_data: dict) -> Tuple:
        """Get evaluation params by optimizing on the given data

        :param epoch_data: dictionary of various values in the epoch
        :type epoch_data: dict

        :return: dict containing evaluation parameters
        """
        metrics = self.compute_epoch_metrics(
                epoch_data['predictions'], epoch_data['targets'])
        param_keys = ['recall', 'threshold']
        params = {key: metrics[key] for key in param_keys}
        return params

    def compute_epoch_metrics(
            self, predictions: Any, targets: Any,
            threshold: float = None, recall: float = 0.9,
            as_logits: bool = True, classes: List[str] = None) -> dict:
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
        :type as_logits: bool, defaults to True
        :param classes: list of classes in the target
        :type classes: List[str], defaults to None

        :return: dictionary of metrics as provided in the config file
        """
        if as_logits:
            # convert logits to probabilities using sigmoid/softmax
            if len(predictions.shape) == 1:
                predictions = torch.sigmoid(predictions)
            elif (len(predictions.shape) == 2 and predictions.shape[1] == 1):
                predictions = torch.sigmoid(predictions.squeeze())
            else:
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
            metrics['roc-curve'] = plot_classification_metric_curve(
                fprs, tprs, xlabel='False Positive Rate',
                ylabel='True Positive Rate')
            plt.close()

            if len(torch.unique(targets)) ==  1:
                metrics['auc-roc'] = '-'
            else:
                metrics['auc-roc'] = roc_auc_score(targets, predict_proba)

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

    def save(self, epoch_metric_values: Dict, use_wandb: bool):
        """Saves the model and optimizer states

        :param epoch_metric_values: validation metrics computed for current epoch
        :type epoch_metric_values: Dict
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        """

        # updating the best metric and obtaining save-related metadata
        save_status = self.checkpoint.update_best_metric(
            self.epoch_counter, epoch_metric_values)

        # if save status indicates necessaity to save, save the model
        # keeping this part model-class dependent since this can change
        # with models
        if save_status['save']:
            logging.info(color(save_status['info'], 'red'))
            torch.save({
                'network': self.network.get_state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch_counter,
                'metrics': epoch_metric_values
            }, save_status['path'])

    def load(self, load_config: Dict):
        """Defines and executes logic to load checkpoint for fine-tuning.

        :param load_config: config defining parameters related to
            loading the model and optimizer
        :type load_config: Dict
        """
        if load_config['version'] is not None:
            load_dir = join(
                self.config.paths['OUT_DIR'], load_config['version'],
                'checkpoints')
            self.load_path = get_saved_checkpoint_path(
                load_dir, load_config['load_best'], load_config['epoch'])

            logging.info(color("=> Loading model weights from {}".format(
                self.load_path)))
            if exists(self.load_path):
                checkpoint = torch.load(self.load_path)
                self.network.load_state_dict(checkpoint['network'])

                if load_config['resume_optimizer']:
                    logging.info(color("=> Resuming optimizer params"))
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

                if load_config['resume_epoch']:
                    self.epoch_counter = checkpoint['epoch']
            else:
                sys.exit(color('Checkpoint file does not exist at {}'.format(
                    self.load_path), 'red'))

    def _accumulate_lr(self, learning_rates: List[Any]) -> dict:
        """Accumulate learning rate values

        :param learning_rates: Dynamically accumulated learning rates per batch
            over all epochs
        :type learning_rates: List[Any]
        :return: dict containing a running list of learning rates
        """
        learning_rates.append(self.optimizer.param_groups[0]['lr'])
        return learning_rates

    def process_batch(self, batch: Any, mode: str = None) -> Tuple[Any, Any]:
        """Returns the predictions and targets for each batch

        :param batch: one batch of data containing inputs and targets
        :type batch: Any
        :param mode: train/val/test mode
        :type mode: str

        :return: dict containing predictions and targets
        """
        inputs = batch['signals'].to(self.device)
        labels = batch['labels'].to(self.device)

        if mode is not None and 'train' in mode:
            predictions = self.network(inputs)
        else:
            with torch.no_grad():
                predictions = self.network(inputs)

        batch_data = {
            'inputs': inputs,
            'predictions': predictions,
            'targets': labels,
            'items': batch['items']
        }

        return batch_data

    def _update_wandb(self, mode: str, epoch_losses: dict, metrics: dict,
                      epoch_data: dict, learning_rates: List[Any] = None,
                      batch_losses: defaultdict = None):
        """Logs values to wandb

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
        :type learning_rates: List[Any], defaults to None
        :param batch_losses: Dynamically accumulated losses per batch
        :type batch_losses: defaultdict, defaults to None
        """
        super(ClassificationModel, self)._update_wandb(
            mode, epoch_losses, metrics)

        # decide indices to visualize
        indices = get_indices(epoch_data['targets'])

        # log data summary only once at the first epoch
        if not self.epoch_counter:
            f, ax = plt.subplots(1)
            ax.hist(epoch_data['targets'])
            ax.set_xticks(np.unique(epoch_data['targets']))
            wandb.log({
                '{}/data_distribution'.format(mode): wandb.Image(fig2im(f))
            }, step=self.epoch_counter)
            plt.close()

        # log audios
        audios = get_audios(
            epoch_data['items'][indices], F.softmax(
                epoch_data['predictions'][indices], -1),
            epoch_data['targets'][indices])
        self.wandb_logs['{}/audios'.format(mode)] = audios

        # log learning rates vs losses
        if learning_rates is not None and batch_losses is not None:
            lr_vs_loss = plot_lr_vs_loss(
                learning_rates, batch_losses['loss'], as_figure=True)
            self.wandb_logs['{}/lr-vs-loss'.format(mode)] = lr_vs_loss
            plt.close()

        # log original inputs if they are not raw waveform
        if not self.data_config['raw_waveform']:
            input_images = get_images(
                epoch_data['inputs'][indices].tolist(),
                epoch_data['predictions'][indices],
                epoch_data['targets'][indices])
            self.wandb_logs['{}/inputs'.format(mode)] = input_images

        self.wandb_logs['{}/confusion_matrix'.format(mode)] = wandb.Image(
            get_confusion_matrix(metrics['confusion_matrix'],
                                 self.model_config['classes']))

        self.wandb_logs['{}/pr-curve'.format(mode)] = metrics['pr-curve']
        self.wandb_logs['{}/roc-curve'.format(mode)] = metrics['roc-curve']
        self.wandb_logs['{}/ss-curve'.format(mode)] = metrics['ss-curve']

        # log to wandb
        wandb.log(self.wandb_logs, step=self.epoch_counter)

    def evaluate(
            self, data_loader: DataLoader, mode: str, use_wandb: bool = True,
            ignore_cache: bool = True, threshold: float = None,
            recall: float = 0.9, data_only: bool = False,
            save: bool = True, log_summary: bool = True):
        """Evaluate the model on given data

        :param data_loader: data_loader made from the evaluation dataset
        :type data_loader: DataLoader
        :param mode: split of the data represented by the dataloader (train/test/val)
        :type mode: str
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool, defaults to True
        :param ignore_cache: whether to ignore cached values
        :type ignore_cache: bool, defaults to True
        :param threshold: confidence threshold to be used for binary
            classification; if None, the optimal threshold is found.
        :type threshold: float, defaults to None
        :param recall: minimum recall to choose the optimal threshold
        :type recall: float, defaults to 0.9
        :param data_only: whether to only return the epoch data without
            computing metrics
        :type data_only: bool, defaults to False
        :param save: whether to save eval data
        :type save: bool, defaults to True
        :param log_summary: whether to log epoch summary
        :type log_summary: bool, defaults to True
        """
        ckpt_name = basename(self.load_path).split('.')[0]
        cache_path = join(
            self.config.output_dir, 'evaluation', ckpt_name, 'cache', '{}.pt'.format(mode))
        makedirs(dirname(cache_path), exist_ok=True)

        # load cache if `ignore_cache=False`
        if exists(cache_path) and not ignore_cache:
            logging.info(
                color('Using cached values from {}'.format(cache_path), 'red'))
            results = torch.load(cache_path)
        else:
            logging.info(color('Ignoring cache', 'red'))
            # prevent logging anything into wandb when processing epoch
            results = self.process_epoch(
                data_loader, mode=mode, training=False, use_wandb=False,
                log_summary=log_summary)

            logging.info(
                color('Saving cached values to {}'.format(cache_path), 'red'))
            torch.save(results, cache_path)

        if data_only:
            return results

        predictions = results['predictions']
        targets = results['targets']

        threshold = threshold if threshold is not None else results['threshold']

        logging.info(color("Using threshold: {}".format(threshold)))
        metrics = self.compute_epoch_metrics(
            predictions, targets, threshold=threshold, recall=recall)
        logging.info(metrics)

        # save evaluation data
        save_dir = join(self.config.output_dir, 'evaluation', ckpt_name)

        # update wandb
        if use_wandb:
            self._update_wandb(mode, {}, metrics, results)

        if save:
            logging.info(color('Saving eval data', 'red'))
            _save_eval_data(
                save_dir, mode, results['items'], predictions, targets,
                metrics)


class ClassificationModelBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs):
        return ClassificationModel(**kwargs)
