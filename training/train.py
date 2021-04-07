"""
Trainer script.

Example:
To set number of CPUs to 10:
>>> taskset --cpu-list 0-10 python train.py -v experiments/resnet18-coughnet.yml --wandb_entity wadhwani

To use all CPUs:
>>> python train.py -v experiments/resnet18-coughnet.yml --wandb_entity wadhwani

usage: train.py [-h] -v VERSION [-n NUM_WORKERS] [--debug] [-o] [--resume]
                [--id ID] [--no-wandb] [--seed SEED]
                [--wandb_entity WANDB_ENTITY]

Trains a model

optional arguments:
  -h, --help            show this help message and exit
  -v VERSION, --version VERSION
                        path to the experiment config file
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        number of CPU workers to use
  --debug               specify where a debugging run
  -o, --overfit-batch   specify whether the run is to test overfitting on a
                        batch
  --resume              whether to resume experiment in wandb
  --id ID               experiment ID in wandb
  --no-wandb            whether to ignore using wandb
  --seed SEED           seed for the experiment
  --wandb_entity WANDB_ENTITY
                        your wandb account name
"""
import warnings
import logging
import argparse
import os
from os.path import join, dirname
import multiprocessing as mp
import wandb
from cac.config import Config
from cac.models import factory as model_factory
from cac.utils.logger import set_logger
from training.utils import seed_everything

warnings.simplefilter('ignore')


def train(config, debug, overfit_batch, use_wandb):
    model = model_factory.create(config.model['name'], **{'config': config})
    model.fit(debug=debug, overfit_batch=overfit_batch, use_wandb=use_wandb)


def main(args):
    seed_everything(args.seed)
    config = Config(args.version)

    set_logger(join(config.log_dir, 'train.log'))
    logging.info(args)
    os.environ['WANDB_ENTITY'] = args.wandb_entity
    os.environ['WANDB_PROJECT'] = "cough-against-covid"
    os.environ['WANDB_DIR'] = dirname(config.checkpoint_dir)

    run_name = args.version.replace('/', '_')
    wandb.init(name=run_name, dir=dirname(config.checkpoint_dir),
               notes=config.description, resume=args.resume,
               id=args.id)
    wandb.config.update(
        config.__dict__,
        allow_val_change=config.allow_val_change)

    config.num_workers = args.num_workers
    train(config, args.debug, args.overfit_batch, args.no_wandb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a model")
    parser.add_argument('-v', '--version', required=True, type=str,
                        help='path to the experiment config file')
    parser.add_argument('-n', '--num_workers', default=mp.cpu_count(), type=int,
                        help='number of CPU workers to use')
    parser.add_argument('--debug', action='store_true',
                        help='specify where a debugging run')
    parser.add_argument('-o', '--overfit-batch', action='store_true',
                        help='specify whether the run is to test overfitting on a batch')
    parser.add_argument('--resume', action='store_true',
                        help='whether to resume experiment in wandb')
    parser.add_argument('--id', type=str, default=None,
                        help='experiment ID in wandb')
    parser.add_argument('--no-wandb', action='store_false',
                        help='whether to ignore using wandb')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for the experiment')
    parser.add_argument('--wandb_entity', default='wadhwani', type=str,
                        help='your wandb account name')
    args = parser.parse_args()
    main(args)
