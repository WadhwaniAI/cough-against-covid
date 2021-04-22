"""Contains functions for loading data"""
# pylint: disable=no-member
import logging
from functools import partial
from collections import defaultdict
from typing import Tuple, Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from cac.data import factory as dataset_factory
from cac.data.base import BaseDataset
from cac.data.sampler import sampler_factory
from cac.data.transforms import DataProcessor, annotation_factory
from cac.utils.logger import color

def context_classification_collate(
        batch: Tuple[Dict], zero_pad: bool = False,
        stack: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for classification model to handle variable-length signals

    :param batch: A tuple of dicts of processed signals and the corresponding labels
    :type batch: Tuple[Dict]
    :param zero_pad: whether to zero pad each input to maximum length sequence
        in a batch, defaults to False
    :type zero_pad: bool
    :param stack: whether to stack inputs on dim 0 or not, if inputs are variable lengths
        and `zero_pad=False`, `stack=False` is mandatory
    :type stack: bool
    :returns: A dict containing:
        1) tensor, batch of processed signals zero-padded and stacked on their 0 dim
        2) tensor, batch of corresponding labels
    """
    signals = []
    labels = []
    items = []

    for data_point in batch:
        signal = data_point['signal']

        if zero_pad:
            # transposing here to make it the right shape for zero padding
            # last dimension is assumed to represent timesteps
            signal = signal.transpose(0, -1)

        signals.append(signal)
        labels.append(data_point['label'])
        items.append(data_point['item'])

    if zero_pad:
        # zero pad the list of sequences to the length of the longest
        # sequence and permute the dimensions to match the shape orientation
        # pre zero-padding.
        signals = pad_sequence(signals, batch_first=True)

        # transposing shape back to timsteps-last
        signals = signals.transpose(1, -1)
    else:
        if stack:
            signals = torch.stack(signals)

    collated_batch = {
        'signals': signals.squeeze(),
        'labels': torch.Tensor(labels),
        'items': items
    }

    return collated_batch


def classification_collate(
        batch: Tuple[Dict], zero_pad: bool = False,
        stack: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for classification model to handle variable-length signals

    :param batch: A tuple of dicts of processed signals and the corresponding labels
    :type batch: Tuple[Dict]
    :param zero_pad: whether to zero pad each input to maximum length sequence
        in a batch, defaults to False
    :type zero_pad: bool
    :param stack: whether to stack inputs on dim 0 or not, if inputs are variable lengths
        and `zero_pad=False`, `stack=False` is mandatory
    :type stack: bool
    :returns: A dict containing:
        1) tensor, batch of processed signals zero-padded and stacked on their 0 dim
        2) tensor, batch of corresponding labels
        3) list, paths to the audio files
    """
    signals = []
    labels = []
    items = []

    for data_point in batch:
        signal = data_point['signal']

        if zero_pad:
            # transposing here to make it the right shape for zero padding
            # last dimension is assumed to represent timesteps
            signal = signal.transpose(0, -1)

        signals.append(signal)
        labels.append(data_point['label'])
        items.append(data_point['item'])

    if zero_pad:
        # zero pad the list of sequences to the length of the longest
        # sequence and permute the dimensions to match the shape orientation
        # pre zero-padding.
        signals = pad_sequence(signals, batch_first=True)

        # transposing shape back to timsteps-last
        signals = signals.transpose(1, -1)
    else:
        if stack:
            signals = torch.stack(signals)

    collated_batch = {
        'signals': signals,
        'labels': torch.Tensor(labels),
        'items': items
    }

    return collated_batch


def unsupervised_collate(
        batch: Tuple[Dict], zero_pad: bool = True,
        stack: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for unsupervised learning models to handle variable-length signals
    or variable labels across datasets
    
    TODO: Use https://pytorch.org/docs/stable/data.html#disable-automatic-batching
        for better way of loading entire dataset

    :param batch: A tuple of dicts of processed signals and the corresponding labels
    :type batch: Tuple[Dict]
    :param zero_pad: whether to zero pad each input to maximum length sequence
        in a batch, defaults to True
    :type zero_pad: bool
    :param stack: whether to stack inputs on dim 0 or not, if inputs are variable lengths
        and `zero_pad=False`, `stack=False` is mandatory
    :type stack: bool
    :returns: A dict containing:
        1) tensor, batch of processed signals zero-padded and stacked on their 0 dim
        2) tensor, batch of corresponding labels
        3) list, paths to the audio files
    """
    signals = []
    labels = []
    items = []

    for data_point in batch:
        signal = data_point['signal']

        if zero_pad:
            # transposing here to make it the right shape for zero padding
            # last dimension is assumed to represent timesteps
            signal = signal.transpose(0, -1)

        signals.append(signal)
        labels.append(data_point['label'])
        items.append(data_point['item'])

    if zero_pad:
        # zero pad the list of sequences to the length of the longest
        # sequence and permute the dimensions to match the shape orientation
        # pre zero-padding.
        signals = pad_sequence(signals, batch_first=True)

        # transposing shape back to timsteps-last
        signals = signals.transpose(1, -1)
    else:
        if stack:
            signals = torch.stack(signals)

    collated_batch = {
        'signals': signals,
        'labels': labels,
        'items': items
    }

    return collated_batch


def multitask_collate(
        batch: Tuple[Dict], num_tasks: int, zero_pad: bool = True,
        stack: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for multi-task learning models to handle variable-length signals
    or variable labels across datasets

    :param batch: A tuple of dicts of processed signals and the corresponding labels
    :type batch: Tuple[Dict]
    :param tasks: number of tasks for multi-task learning
    :type tasks: int
    :param zero_pad: whether to zero pad each input to maximum length sequence
        in a batch, defaults to True
    :type zero_pad: bool
    :param stack: whether to stack inputs on dim 0 or not, if inputs are variable lengths
        and `zero_pad=False`, `stack=False` is mandatory
    :type stack: bool
    :returns: A dict containing:
        1) tensor, batch of processed signals zero-padded and stacked on their 0 dim
        2) tensor, batch of corresponding labels
        3) list, paths to the audio files
    """
    signals = []
    labels = []
    items = []

    for _ in range(num_tasks):
        labels.append([])

    for data_point in batch:
        signal = data_point['signal']

        if zero_pad:
            # transposing here to make it the right shape for zero padding
            # last dimension is assumed to represent timesteps
            signal = signal.transpose(0, -1)

        signals.append(signal)

        for index, label in enumerate(data_point['label']):
            labels[index].append(label)

        items.append(data_point['item'])

    if zero_pad:
        # zero pad the list of sequences to the length of the longest
        # sequence and permute the dimensions to match the shape orientation
        # pre zero-padding.
        signals = pad_sequence(signals, batch_first=True)

        # transposing shape back to timsteps-last
        signals = signals.transpose(1, -1)
    else:
        if stack:
            signals = torch.stack(signals)

    for index in range(num_tasks):
        labels[index] = torch.Tensor(labels[index])

    collated_batch = {
        'signals': signals,
        'labels': labels,
        'items': items
    }

    return collated_batch


def get_dataloader(
        cfg: Dict, mode: str, batch_size: int,
        num_workers: int = 10, shuffle: bool = True, drop_last: bool = True
        ) -> Tuple[DataLoader, BaseDataset]:
    """Creates the DataLoader and Dataset objects

    :param cfg: config specifying the various options for creating the
        dataloader
    :type cfg: Dict
    :param mode: mode/split to load; one of {'train', 'test', 'val'}
    :type mode: str
    :param batch_size: number of instances in each batch
    :type batch_size: int
    :param num_workers: number of cpu workers to use, defaults to 10
    :type num_workers: int
    :param shuffle: whether to shuffle the data, defaults to True
    :type shuffle: bool, optional
    :param drop_last: whether to include last batch containing sample
        less than the batch size, defaults to True
    :type drop_last: bool, optional
    :returns: A tuple containing the DataLoader and Dataset objects
    """
    logging.info(color('Creating {} DataLoader'.format(mode), 'blue'))

    # define target transform
    target_transform = None
    if 'target_transform' in cfg:
        target_transform = annotation_factory.create(
            cfg['target_transform']['name'],
            **cfg['target_transform']['params'])

    # define signal transform
    signal_transform = None
    if 'signal_transform' in cfg:
        signal_transform = DataProcessor(cfg['signal_transform'][mode])

    # define Dataset object
    dataset_params = cfg['dataset']['params'].get(mode, {})

    dataset_params.update({
        'target_transform': target_transform,
        'signal_transform': signal_transform,
        'mode': mode,
        'dataset_config': cfg['dataset']['config']
    })

    dataset = dataset_factory.create(cfg['dataset']['name'], **dataset_params)

    # to load entire dataset in one batch
    if batch_size == -1:
        batch_size = len(dataset)

    # define sampler
    sampler_cfg = cfg['sampler'].get(mode, {'name': 'default'})
    sampler_params = sampler_cfg.get('params', {})
    sampler_params.update({
        'target_transform': target_transform,
        'dataset': dataset,
        'shuffle': shuffle
    })
    sampler = sampler_factory.create(sampler_cfg['name'], **sampler_params)

    # define DataLoader object
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(eval(cfg['collate_fn']['name']),
                           **cfg['collate_fn']['params']),
        drop_last=drop_last,
        pin_memory=True)

    return dataloader, dataset
