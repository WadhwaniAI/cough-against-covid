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
from cac.data.dataloader import get_dataloader_multi_signal
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
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from cac.networks.backbones.resnet import resnet18
np.set_printoptions(suppress=True)
from cac.models.utils import get_subsets
from cac.networks.backbones.utils import _correct_state_dict
from cac.networks.multi_signal import * 
from cac.models.classification import ClassificationModel

class MultiSignalClassificationModel(ClassificationModel):
    """Classification model class

    Args:
    :param config: Config object
    :type config: Config
    """
    def __init__(self, config):
        super(ClassificationModel, self).__init__(config)

    def _setup_network(self):
        """Setup the network which needs to be trained"""
        network_name = self.network_config['network_name']
        params = self.network_config['params']
        self.network = eval(f'{network_name}(**{params})').to(self.device)
        logging.info(color(f"Building the network: {type(self.network).__name__}"))

    def _freeze_layers(self):
        """Freeze layers based on config during training"""
        logging.info(color('Freezing specified layers'))
        # Add Stuff Here
 
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
            train_dataloader, _ = get_dataloader_multi_signal(
                self.data_config, 'train',
                self.model_config['batch_size'],
                num_workers=self.config.num_workers,
                shuffle=shuffle,
                drop_last=False)

        # ignore val operations when overfitting on a batch
        if not overfit_batch:
            val_dataloader, _ = get_dataloader_multi_signal(
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
            wandb.watch(self.network)

        best_metric_values = None
        
        # run val epoch before any training
        val_results = self.process_epoch(
            val_dataloader, 'val', training=False, use_wandb=use_wandb
        )
        self.epoch_counter += 1

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
                model_state_dict = self.network.state_dict()
                state_corrected_dict = _correct_state_dict(checkpoint['network'], model_state_dict)
                self.network.load_state_dict(state_corrected_dict, strict=False)

                if load_config['resume_optimizer']:
                    logging.info(color("=> Resuming optimizer params"))
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                
                if load_config['resume_epoch']:
                    self.epoch_counter = checkpoint['epoch']
            else:
                sys.exit(color('Checkpoint file does not exist at {}'.format(
                    self.load_path), 'red'))        

    def process_batch(self, batch: Any, mode: str = None) -> Tuple[Any, Any]:
        """Returns the predictions and targets for each batch

        :param batch: one batch of data containing inputs and targets
        :type batch: Any
        :param mode: train/val/test mode
        :type mode: str

        :return: dict containing predictions and targets
        """
        inputs = []
        for i in range(len(batch['signals'])):
            inputs.append(batch['signals'][i].to(self.device))
        text_signals = batch['context-signals'].float().to(self.device)
        labels = batch['labels'].to(self.device)
        if mode is not None and 'train' in mode:
            predictions = self.network(inputs, text_signals.squeeze())
        else:
            with torch.no_grad():
                predictions = self.network(inputs, text_signals.squeeze())

        batch_data = {
            'inputs': inputs[0],
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

        self.wandb_logs['{}/confusion_matrix'.format(mode)] = wandb.Image(
            get_confusion_matrix(metrics['confusion_matrix'],
                                 self.model_config['classes']))

        self.wandb_logs['{}/pr-curve'.format(mode)] = metrics['pr-curve']
        self.wandb_logs['{}/roc-curve'.format(mode)] = metrics['roc-curve']
        self.wandb_logs['{}/ss-curve'.format(mode)] = metrics['ss-curve']

        # log to wandb
        wandb.log(self.wandb_logs, step=self.epoch_counter)

class MultiSignalClassificationModelBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs):
        return MultiSignalClassificationModel(**kwargs)
