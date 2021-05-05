"""Script for evaluation of ensemble of N models on given dataset.

Example:

$ cfg1=experiments/covid-detection/v9_7_cough_adam_1e-4.yml
$ cfg2=experiments/iclrw/context/v9.7/context-neural.yml
$ python ensemble.py -c experiments/ensemble/cough_context_v9.7.yml

TODO: Add ignore_cache option, use cache when available to save time
"""
import os
from os.path import splitext, join, basename, exists
import argparse
from collections import defaultdict
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cac.config import Config
from cac.utils.logger import set_logger, color
from cac.data.dataloader import get_dataloader
from cac.models import factory as model_factory
from cac.utils.helper import get_from_dict, set_in_dict, individual_level_aggregation

from cac.utils.io import read_yml
from cac.utils.file import repo_path

class EnsembleOnDataset:
    """Class for inference of ensemble of models on a given dataset."""
    def __init__(self, all_model_args, data_args):
        super(EnsembleOnDataset, self).__init__()
        self.all_model_args = all_model_args
        self.data_args = data_args

        # load model logs
        self.all_model_logs = dict()
        for model, args in self.all_model_args.items():
            self.all_model_logs[model] = self._load_model_logs(args, data_args['mode'])
        
        # load ensembling weights
        self.weights = {m: all_model_args[m]['weight'] for m in all_model_args}

        # need model only for model.compute_epoch_metrics
        dummy_version = "experiments/covid-detection/v9_7_cough_adam_1e-4.yml"
        dummy_config = Config(dummy_version)
        metrics_to_track = ['auc-roc', 'specificity', 'recall', 'threshold']
        dummy_config.metrics_to_track = metrics_to_track
        self.dummy_model = self.load_model(dummy_config)

        # compute final metrics
        results = self._ensemble(self.all_model_logs, self.weights)

        output_str = color(f"Results for ensembled model {self.weights} on dataset {self.data_args}: \n", 'green')
        output_str += str({k:results[k] for k in metrics_to_track})

        print(output_str)
    
    def _ensemble(self, all_model_logs, weights):
        # computing average threshold
        threshold = self._weighted_average_across_models(
            all_model_logs, weights, 'threshold'
        )
        predictions = self._weighted_average_across_models(
            all_model_logs, weights, 'predictions'
        )
        targets = self._weighted_average_across_models(
            all_model_logs, weights, 'targets'
        )

        print(color("=> Computing metrics"))
        metrics = self.dummy_model.compute_epoch_metrics(
            predictions, targets,
            as_logits=False, threshold=threshold
        )
        
        return metrics

    def _weighted_average_across_models(self, all_model_logs, weights, key):
        quantity_to_avg = None
        for model in all_model_logs:
            if quantity_to_avg is None:
                quantity_to_avg = all_model_logs[model][key] * weights[model]
            else:
                quantity_to_avg += all_model_logs[model][key] * weights[model]
        return quantity_to_avg

    def _load_model_logs(self, model_args, mode):
        version = splitext(model_args['version'])[0]
        print(color(f"=> Loading logs for {version}"))
        user = model_args['user']
        output_folder = '/output' if user is None else f'/all-output/{user}'
        logs_folder = join(
            output_folder, version + '_inference', 'logs', 'epochwise', mode
        )
        log_path = join(
            logs_folder, f'{model_args["epoch"]}_{model_args["agg_method"]}_aggregated.pt'
        )
        assert exists(log_path), f"logs do not exist for {model_args}. "\
            "Please run inference.py to first generate logs."

        return torch.load(log_path)

    def load_model(self, config):
        model = model_factory.create(config.model['name'], **{'config': config})
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates a model")
    parser.add_argument('-c', '--metaconfig', required=True, type=str,
                        help='meta config containing info about models to ensemble')
    args = parser.parse_args()

    meta_cfg_path = join(repo_path, 'configs', args.metaconfig)
    assert exists(meta_cfg_path), f"Meta-config file does not exists at {meta_cfg_path}"
    meta_cfg = read_yml(meta_cfg_path)

    ensemble = EnsembleOnDataset(meta_cfg['models'], meta_cfg['data'])
