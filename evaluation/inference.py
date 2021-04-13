"""Script for inference for a given config/user/epoch on given dataset.

Example:

$ cfg=cough-clf/mixup/wiai-v9.4/resnet18/base-best-step-scheduler.yml
$ python inference.py -v $cfg -u piyush -e 100 -dv v9.4 -m test --at softmax

TODO: Add ignore_cache option, use cache when available to save time
"""
import os
from os.path import splitext, join, basename
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


class InferenceOnDataset:
    """Class for inference of a model on a given dataset."""
    def __init__(self, args):
        super(InferenceOnDataset, self).__init__()

        # load config
        self.print_update("Loading and setting config")
        config = self.load_and_set_config(args)

        # set logging
        set_logger(join(config.log_dir, 'inference.log'))

        # load optimal threshold for val set on this epoch
        epoch_val_logs = self.load_epoch_logs(config.log_dir, args.epoch)
        if args.threshold is None:
            args.threshold = epoch_val_logs['threshold']

        # load data
        self.print_update("Loading data")
        dataloader = self.load_data(config, args)

        # load model
        self.print_update("Loading model")
        model = self.load_model(config)

        # forward pass
        self.print_update("Running forward pass")
        results = self.forward_pass(model, dataloader, args.mode, args.threshold)
        # save logs: individual predictions
        self.save_logs(results, config.log_dir_inference, args.mode, fname="0.pt")

        # individual level aggregation
        self.print_update("individual level aggregation")
        ila_results = self.results_post_aggregation(
            model, results, epoch_val_logs, args.agg_method, args.mode, at=args.at
        )
        # save logs: individual predictions post aggregation
        fname = f"0_{args.agg_method}_aggregated.pt"
        self.save_logs(ila_results, config.log_dir_inference, args.mode, fname=fname)

    def load_and_set_config(self, args):
        config = Config(args.version, user=args.user)

        # add `test` field copying from `val`
        config.data = self.add_test_mode_to_config(
            config.data, [['dataset', 'params'], ['sampler'], ['signal_transform']]
        )
        config.model = self.add_test_mode_to_config(
            config.model, [['eval', 'aggregate'], ['loss'], ['subset_tracker']]
        )

        # add info about the dataset
        config.data['dataset']['config'] = [
            {
                "name": args.data_name,
                "version": args.data_version,
                "mode": args.mode
            }
        ]

        # add info about the model
        config.model['load']['epoch'] = args.epoch
        config.model['load']['load_best'] = args.load_best
        config.model['load']['version'] = args.version.replace(".yml", "")

        # set inference directories for logging
        dirpaths = ['config_save_path', 'output_dir', 'log_dir', 'checkpoint_dir']
        for key in dirpaths:
            train_version = splitext(args.version)[0]
            infer_version = train_version + "_inference"

            dirpath = getattr(config, key).replace(train_version, infer_version)
            os.makedirs(dirpath, exist_ok=True)
            setattr(config, key + '_inference', dirpath)

        # set metrics to track
        config.metrics_to_track = ['auc-roc', 'specificity', 'recall', 'threshold']

        return config

    def load_data(self, config, args):
        dataloader, _ = get_dataloader(
            config.data, args.mode, config.model['batch_size'],
            num_workers=args.num_workers, shuffle=False, drop_last=False
        )
        return dataloader

    def load_model(self, config):
        model = model_factory.create(config.model['name'], **{'config': config})
        return model

    def forward_pass(self, model, dataloader, mode, threshold):
        results = model.process_epoch(
            dataloader, mode=mode, use_wandb=False, threshold=threshold
        )
        results['args'] = args
        return results

    def save_logs(self, results, logdir, mode, fname):
        logdir = join(logdir, 'epochwise', mode)
        os.makedirs(logdir, exist_ok=True)

        assert fname.endswith(".pt")
        torch.save(results, join(logdir, fname))

    def results_post_aggregation(self, model, results, val_results, agg_method, mode, at):

        # obtain optimal threshold based on val set
        val_ila_results = individual_level_aggregation(
            val_results['paths'], val_results['targets'],
            val_results['predictions'], agg_method=agg_method, at=at
        )
        val_metrics = model.compute_epoch_metrics(
            val_ila_results['predictions'], val_ila_results['targets'],
            as_logits=False, threshold=None
        )
        agg_threshold = val_metrics['threshold']

        # display results
        metric_log = "V: {} | ILA: Val | {}".format(
            model.config.version, mode.capitalize()
        )
        for metric in model.config.metrics_to_track:
            metric_log += ' | {}: {:.4f}'.format(metric, val_metrics[metric])
        print(color(metric_log, 'green'))

        # compute ILA results on given mode using obtained threshold
        ila_results = individual_level_aggregation(
            [x.path for x in results['items']], results['targets'],
            results['predictions'], agg_method=agg_method
        )
        metrics = model.compute_epoch_metrics(
            ila_results['predictions'], ila_results['targets'],
            as_logits=False, threshold=agg_threshold
        )
        ila_results.update(metrics)

        # display results
        metric_log = "V: {} | ILA: Test | {}".format(
            model.config.version, mode.capitalize()
        )
        for metric in model.config.metrics_to_track:
            metric_log += ' | {}: {:.4f}'.format(metric, metrics[metric])
        print(color(metric_log, 'green'))


        return ila_results

    @staticmethod
    def load_epoch_logs(logdir, epoch, mode="val"):
        epoch_logfile = join(logdir, f"epochwise/{mode}/{epoch}.pt")
        epoch_logs = torch.load(epoch_logfile)
        return epoch_logs

    # helper functions specifc to this class
    @staticmethod
    def add_test_mode_to_config(
        _dict, edit_keys, new_mode="test", existing_mode="val"
        ):
        for keys in edit_keys:
            existing_mode_values = get_from_dict(_dict, keys + [existing_mode])
            set_in_dict(_dict, keys + [new_mode], existing_mode_values)
        return _dict

    @staticmethod
    def print_update(update, sep="=", numsep=30):
        string = "{} | {} | {}".format(sep * numsep, update.upper(), sep * numsep)
        print(color(string, "white"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates a model")

    # model-related inputs
    parser.add_argument('-v', '--version', required=True, type=str,
                        help='path to the experiment config file')
    parser.add_argument('-u', '--user', required=True, type=str,
                        help='specifies the user owning the checkpoint')
    parser.add_argument('-e', '--epoch', type=int, default=-1,
                        help='specifies the checkpoint epoch to load')
    parser.add_argument('-b', '--load_best', action='store_true',
                        help='whether to load the best saved checkpoint')
    parser.add_argument('-t', '--threshold', default=None,
                        help='specifies the confidence threshold to compute labels')
    parser.add_argument('-a', '--agg_method', default='max', type=str,
                        help='method of individual level aggregation')
    parser.add_argument('--at', default='max', type=str, required=True,
                        choices=['softmax', 'sigmoid'], help='position to apply aggregation')

    # data-related inputs
    parser.add_argument('-dn', '--data_name', type=str, default='wiai-facility',
                        help='specifies the name of data to evaluate on')
    parser.add_argument('-dv', '--data_version', type=str, default='v9.4',
                        help='specifies the version of data to evaluate on')
    parser.add_argument('-m', '--mode', type=str, default='test',
                        help='specifies the split of data to evaluate on')

    # other inputs
    parser.add_argument('-n', '--num_workers', default=10, type=int,
                        help='number of CPU workers to use')
    parser.add_argument('--no-wandb', action='store_false',
                        help='whether to ignore using wandb')
    parser.add_argument('-i', '--ignore-cache', action='store_true',
                        help='whether to ignore cache')

    args = parser.parse_args()

    inference = InferenceOnDataset(args)
