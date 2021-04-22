"""
Defines ContextClassificationDataset class which is used for classification tasks
where each input has a single output.
"""
from os.path import join
from typing import Tuple, List, Union
import pandas as pd

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from cac.data.audio import AudioItem
from cac.data.base import BaseDataset
from cac.data.transforms import DataProcessor, ClassificationAnnotationTransform
from cac.data.utils import read_dataset_from_config
from cac.utils.typing import DatasetConfigDict
from torch.utils.data import Dataset

class ContextClassificationDataset(Dataset):
    """Dataset class for single-label Text classification

    :param dataset_config: defines the config for the
        data to be loaded. The config is specified by a list of dict, with each dict
        representing: (dataset_name, dataset_version, mode [train, test, val])
    :type dataset_config: DatasetConfigDict
    :param target_transform: defines the transformation
        to be applied on the raw targets to make them processable.
    :type target_transform: ClassificationAnnotationTransform
    :param signal_transform: Will not be used, added to use the get_dataloader() without changes
    :type signal_transform: DataProcessor
    :param features: List of features that are to be used for text classification, 
    (This is hard coded for wiai-facility dataset, for other datasets TODO) 
    :type features: List     
    """
    def __init__(self, dataset_config: List[DatasetConfigDict], features: List[str],
                signal_transform: DataProcessor = None,
                target_transform: ClassificationAnnotationTransform = None,
                attributes_file: str = '/data/wiai-facility/processed/attributes_context_processed.csv'):
        super(ContextClassificationDataset, self).__init__()
        self.dataset_config = dataset_config
        self.target_transform = target_transform
        self.attributes = pd.read_csv(attributes_file)
        self.load_datasets()
        self.features = features
    
    def load_datasets(self):
        """Load the dataset as specified by self.dataset_config"""
        self.items = []
        is_included_dict = dict()

        for dataset_config in self.dataset_config:
            data_info = read_dataset_from_config(dataset_config)

            for i in tqdm(range(len(data_info['file'])), desc='Loading items'):
                path, label = data_info['file'][i], data_info['label'][i]
                patient_id = self.get_patient_id(path)
                if patient_id in is_included_dict.keys(): continue

                start = data_info['start'][i] if 'start' in data_info.keys() else None
                end = data_info['end'][i] if 'end' in data_info.keys() else None

                audio_item = AudioItem(path=path, label=label, start=start, end=end)
                self.items.append(audio_item)
                is_included_dict[patient_id] = True
        
        patient_ids = [self.get_patient_id(x.path) for x in self.items]
        # sanity check
        assert len([x for x in patient_ids if x not in self.attributes.patient_id.values]) == 0
        self.attributes = self.attributes[self.attributes.patient_id.isin(patient_ids)]

    def __getitem__(
            self, index: int, as_tensor=True
            ) -> Tuple[torch.Tensor, Union[List[str], int]]:
        item = self.items[index]

        self._check_item(item)
        patient_id = self.get_patient_id(item.path)
        signal = torch.from_numpy(
            self.attributes[self.attributes.patient_id == patient_id][self.features].values)

        label = item.label['classification']
        if self.target_transform is not None:
            label = self.target_transform(label)

        instance = {
            'signal': signal.float(),
            'label': label,
            'item': item
        }

        return instance

    def _check_item(self, item: AudioItem):
        assert 'classification' in item.label,\
            "Item at index {} does not contain 'classification' in label".format(index)

    def __len__(self):
        return len(self.items)

    def get_patient_id(self, file):
        s = file.split('/')[-1].split('_')
        patient_id = '_'.join([s[0], s[1]])
        return patient_id

class ContextClassificationDatasetBuilder:
    """Builds a ContextClassificationDataset object"""
    def __call__(self, mode: str, dataset_config: List[dict], **kwargs):
        """Builds a ContextClassificationDataset object

        :param mode: mode/split to load; one of {'train', 'test', 'val'}
        :type mode: str
        :param dataset_config: list of dictionaries, each containing
            (name, version, mode) corresponding to a dataset
        :type dataset_config: List[dict]
        :param **kwargs: dictionary containing values corresponding to the arguments of
            the ContextClassificationDataset class
        :type **kwargs: dict
        :returns: a ContextClassificationDataset object
        """
        for i, config in enumerate(dataset_config):
            dataset_config[i]['mode'] = mode

        kwargs['dataset_config'] = dataset_config
        self._instance = ContextClassificationDataset(**kwargs)
        return self._instance
