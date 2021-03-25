from os import makedirs
from os.path import dirname, join
from collections import defaultdict
import torch
import torch.nn.functional as F
import pandas as pd


def _save_eval_data(save_dir, mode, items, predictions, targets, metrics):
    # log predictions and targets
    data = {
        'items': items,
        'predictions': predictions,
        'targets': targets,
    }
    makedirs(save_dir, exist_ok=True)
    save_path = join(save_dir, '{}_experiment_data.pkl'.format(mode))
    torch.save(data, save_path)

    #  log metrics
    save_path = join(save_dir, '{}_metrics.csv'.format(mode))

    # remove any keys which do not have a 1-d value
    delete_matches = ['confusion_matrix', 'pr-curve']
    for key in list(metrics.keys()):
        if any(match_key in key for match_key in delete_matches):
            metrics.pop(key)
    pd.DataFrame(metrics, index=[0]).to_csv(save_path, index=False)


def aggregate_values(predictions, method='median', dim=-1):
    if method == 'median':
        predictions, _ = torch.median(predictions, dim)
    elif method == 'mean':
        predictions = torch.mean(predictions, dim)
    else:
        raise NotImplementedError(
            'Aggregation support only for median and mean')

    return predictions


def aggregate_epoch_values(all_epoch_values, method):
    predictions = defaultdict(list)
    targets = defaultdict()
    for epoch, epoch_values in all_epoch_values.items():
        for mode, mode_values in epoch_values.items():
            if mode not in targets:
                targets[mode] = mode_values['targets']

            predictions[mode].append(F.softmax(
                mode_values['predictions'], -1)[:, 1])

    for mode, mode_values in predictions.items():
        predictions[mode] = torch.stack(predictions[mode], -1)
        predictions[mode] = aggregate_values(predictions[mode], method)

    return predictions, targets
