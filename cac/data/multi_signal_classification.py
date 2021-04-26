import logging
from functools import partial
from collections import defaultdict
from typing import Tuple, Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from cac.data import factory as dataset_factory
from cac.data.base import BaseDataset
from cac.data.sampler import sampler_factory
from cac.data.transforms import DataProcessor, annotation_factory
from cac.utils.logger import color
from cac.utils.typing import DatasetConfigDict
import pandas as pd

class MultiSignalClassificationDataset(Dataset):
    '''
    Creates Dataset for Multi-Signal Joint Training

    :param signal_wise_data_cfgs: defines the config for the
        data to be loaded. The config is specified by a list of list of dict, with each dict
        representing: (dataset_name, dataset_version, mode [train, test, val])
    :type signal_wise_data_cfgs: List[DatasetConfigDict]
    :param mode: One of these ['train', 'val', 'test']
    :type mode: str
    :param features: List of features that are to be used for text classification, 
    (This is hard coded for wiai-facility dataset, for other datasets TODO) 
    :type features: List 
    :param attribute_file: CSV file to pick up the contextual data
    :type attribute_file: str
    '''
    def __init__(self, signal_wise_data_cfgs: DatasetConfigDict, mode: str = 'train',
                features: List = ['enroll_patient_age'],
                attribute_file: str = '/data/wiai-facility/processed/attributes_context_processed.csv'):
        super(Dataset, self).__init__()
        self.target_transform = None
        self.dataset_compiled = []
        self.attribute_file = attribute_file
        self.features = features

        for sub_config in signal_wise_data_cfgs:
            # define target transform
            if 'target_transform' in sub_config:
                self.target_transform = annotation_factory.create(
                    sub_config['target_transform']['name'],
                    **sub_config['target_transform']['params'])

            # define signal transform
            signal_transform = None
            if 'signal_transform' in sub_config:
                signal_transform = DataProcessor(sub_config['signal_transform'][mode])

            # define Dataset object
            dataset_params = sub_config['dataset']['params'].get(mode, {})

            dataset_params.update({
                'target_transform': self.target_transform,
                'signal_transform': signal_transform,
                'mode': mode,
                'dataset_config': sub_config['dataset']['config']
            })

            self.dataset_compiled.append(dataset_factory.create(sub_config['dataset']['name'], **dataset_params))
        
        # sanity check
        patient_ids_base = [self.get_patient_id(x.path) for x in self.dataset_compiled[0].items]
        for i in range(1, len(self.dataset_compiled)):
            patient_ids = [self.get_patient_id(x.path) for x in self.dataset_compiled[i].items]
            assert all([a == b for a, b in zip(patient_ids_base, patient_ids)])

        patient_ids = [self.get_patient_id(x.path) for x in self.dataset_compiled[0].items]
        self.preprocess_attributes(patient_ids)
        self.items = self.dataset_compiled[0].items

    def __getitem__(
            self, index: int, as_tensor=True):
        signal, label, item = [], [], []
        for dataset in self.dataset_compiled:
            i = dataset[index]
            item.append(i['item'])
            signal.append(i['signal'])
            label.append(i['label'])

        patient_id = self.get_patient_id(self.items[index].path)
        context_signal = torch.from_numpy(self.attributes[self.attributes.patient_id == patient_id][self.features].values)
        
        return {
            'signal' : signal,
            'context-signal' : context_signal,
            'label' : label[0],
            'item' : item[0], # TODO: Add support for multiple items   
        }

    def __len__(self):
        return len(self.items)
    
    def get_patient_id(self, file):
        try:
            s = file.split('/')[-1].split('_')
            patient_id = '_'.join([s[0], s[1]])
        except AssertionError:
            print (f"No Patient Id associated with {file}")
        return patient_id
    
    def preprocess_attributes(self, patient_ids):
        logging.info(color(f'Loading Text Signals From {self.attribute_file} ', 'green'))
        self.attributes = pd.read_csv(self.attribute_file)
        assert len([x for x in patient_ids if x not in self.attributes.patient_id.values]) == 0
        self.attributes = self.attributes[self.attributes.patient_id.isin(patient_ids)]