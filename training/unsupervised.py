import warnings
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


from typing import List
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from cac.utils.viz import fig2im


def refine_attributes(attributes: List[dict]) -> pd.DataFrame:
    """Converts list of dicts of attrbutes into dataframe

    :param attributes: list of dict, each dict contains various attributes
    :type attributes: List[dict]
    """
    df = pd.DataFrame(attributes)

    df['data-source'] = df[['state', 'facility']].apply(lambda x: x[0] + ': {}'.format(x[1]), axis=1)
    df['disease_status'] = df['disease_status'].apply(lambda x: x.upper())
    df['age_bucket'] = pd.cut(df.age, bins=list(np.linspace(0, 100, 10, dtype=int)))

    return df


def scatter2d(x1, x2, row_values_: pd.DataFrame, label: str, legend: bool = True,
              ignore_list: List[dict] = [], title: str = None,
              as_figure: bool = False, annotate: bool = False):
    """Scatter plot function for visualizing embeddings conditioned on attributes

    :param x1: first component to plot
    :type x1: np.ndarray
    :param x2: second component to plot
    :type x2: np.ndarray
    :param row_values_: dataframe of attributes
    :type row_values_: pd.DataFrame
    :param label: attribute based on which clustering of embeddings will be plotted,
        `label` must be a column in `row_values_`
    :type label: str
    :param legend: flag to decide whether to show the legend
    :type legend: bool, defaults to True
    :param ignore_list: list of dicts of form `{key: , 'values': []}`. From the dataframe,
        rows with values of `row_values_[key]`in `values` will be ignored while plotting
    :type ignore_list: List[dict], defaults to []
    :param title: title of the plot
    :type title: str, defaults to None
    :param as_figure: flag to decide whether to return raw figure to be logged
    :type as_figure: bool, defaults to False
    :param annotate: flag to decide whether to show the annotatations for every point
    :type annotate: bool, defaults to False

    :returns vis_image: plt.figure of the scatter plot
    """
    row_values = row_values_.copy()
    
    # check if the label columns exists
    assert label in row_values.columns
    
    # drop where label column is NaN
    row_values.dropna(subset=[label], inplace=True)
    
    # ignore certain values in given columns
    for ignore_dict in ignore_list:
        key, values = ignore_dict['key'], ignore_dict['values']
        row_values = row_values[~row_values[key].isin(values)]
    
    # retaining only relevant indices in latent embeddings
    keep_indices = list(row_values.index)
    x1 = x1[keep_indices]
    x2 = x2[keep_indices]

    labels = row_values[label].values
    unique_labels = np.unique(labels)

    colors = cm.plasma(np.linspace(0, 1, len(unique_labels)))

    figure, ax = plt.subplots(1, figsize=(10, 10))

    for (i, label), color in zip(enumerate(unique_labels), colors):
        indices = np.where(labels == label)
        num = len(indices[0])
        ax.scatter(x1[indices], x2[indices], label='{} : {}'.format(label, num), color=color)

        if annotate:
            for j in indices[0]:
                ax.annotate('P{}'.format(i), (x1[j] + 0.1, x2[j] + 0.1))

    ax.set_ylabel('Component 2')
    ax.set_xlabel('Component 1')
    
    if title is not None:
        ax.set_title(title)

    ax.grid()

    if legend:
        ax.legend(loc='best')

    if as_figure:
        return figure

    vis_image = fig2im(figure)
    plt.close()
    return vis_image


def _update_wandb(results: dict, desc: str):
    """Updates W&B with results

    :param results: dict containing the latent embeddings, attributes etc.
    :type results: dict
    :param desc: description of the experiment
    :type desc: str
    """
    wandb_logs = {}

    X, Z, Y = results['input'], results['latent'], results['labels']

    df = refine_attributes(Y)

    # data-source plot
    figure = scatter2d(Z[:, 0], Z[:, 1], df, label='data-source',
        title='Audio embeddings by data source: {}'.format(desc))
    wandb_logs.update({'overall/data-source': wandb.Image(figure)})
    plt.close()

    labels = ['disease_status', 'gender', 'fever', 'cough', 'shortness_of_breath', 'age_bucket', 'unique_id']
    facilities = list(df['data-source'].unique())

    for label in labels:
        legend = True if label != 'unique_id' else False
        annotate = not legend

        figure = scatter2d(Z[:, 0], Z[:, 1], df, label=label, ignore_list=[],
                  title='Audio embeddings by {}: overall [{}]'.format(label, desc), legend=legend)
        wandb_logs.update({'overall/{}'.format(label): wandb.Image(figure)})

        for facility in facilities:
            ignore_list = [
                {
                    'key': 'data-source',
                    'values': [x for x in facilities if x != facility]
                }
            ]

            figure = scatter2d(Z[:, 0], Z[:, 1], df, label=label, ignore_list=ignore_list,
                      title='Audio embeddings by {}: {} [{}]'.format(label, facility, desc), legend=legend, annotate=annotate)
            wandb_logs.update({'{}/{}'.format(label, facility): wandb.Image(figure)})

    # log to wandb
    wandb.log(wandb_logs, step=1)


def train(config, debug, use_wandb):
    model = model_factory.create(config.model['name'], **{'config': config})
    results = model.fit(mode='all', return_predictions=True)

    # TODO: Use cached values for the same config
    # TODO: Use chunks of audio 1-10 sounds with x chunks per person
    # TODO: Use entire dataset (coughs + breathing sounds + audio 1-10)

    if use_wandb:
        desc = config.description
        _update_wandb(results, desc)


def main(args):
    seed_everything()
    config = Config(args.version)

    set_logger(join(config.log_dir, 'unsupervised.log'))

    if args.no_wandb:
        os.environ['WANDB_ENTITY'] = "wadhwani"
        os.environ['WANDB_PROJECT'] = "cough-against-covid"
        os.environ['WANDB_DIR'] = dirname(config.checkpoint_dir)

        run_name = args.version.replace('/', '_')
        wandb.init(name=run_name, dir=dirname(config.checkpoint_dir),
                   notes=config.description, resume=args.resume,
                   id=args.id)
        wandb.config.update(config.__dict__)

    config.num_workers = args.num_workers
    train(config, args.debug, args.no_wandb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a model")
    parser.add_argument('-v', '--version', required=True, type=str,
                        help='path to the experiment config file')
    parser.add_argument('-n', '--num_workers', default=mp.cpu_count(), type=int,
                        help='number of CPU workers to use')
    parser.add_argument('--debug', action='store_true',
                        help='specify where a debugging run')
    parser.add_argument('--resume', action='store_true',
                        help='whether to resume experiment in wandb')
    parser.add_argument('--id', type=str, default=None,
                        help='experiment ID in wandb')
    parser.add_argument('--no-wandb', action='store_false',
                        help='whether to ignore using wandb')
    args = parser.parse_args()
    main(args)
