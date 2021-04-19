"""Generic python helper functions."""
from functools import reduce
import operator
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn.functional as F

from cac.utils.file import get_unique_id


def get_from_dict(dataDict, mapList):
    """Credits: https://stackoverflow.com/questions/14692690/\
        access-nested-dictionary-items-via-a-list-of-keys"""
    return reduce(operator.getitem, mapList, dataDict)


def set_in_dict(dataDict, mapList, value):
    """Credits: https://stackoverflow.com/questions/14692690/\
        access-nested-dictionary-items-via-a-list-of-keys"""
    get_from_dict(dataDict, mapList[:-1])[mapList[-1]] = value


def individual_level_aggregation(paths, targets, predictions, at='softmax', agg_method="max"):
    """Applies ILA on given individual file level predictions"""
    assert agg_method in ['min', 'max', 'mean', 'median']

    df = pd.DataFrame(None, columns=['path', 'id'])
    df['path'] = paths
    df['id'] = df['path'].apply(get_unique_id)
    id_groups = df.groupby('id').groups

    if at == 'softmax':
        predictions = F.softmax(predictions, -1)
        predictions = predictions[:, 1]
    elif at == 'sigmoid':
        predictions = torch.sigmoid(predictions.squeeze())
    else:
        raise NotImplementedError

    ila_results = defaultdict(list)
    for uid, indices in tqdm(id_groups.items(), f"Aggregating ILA by {agg_method}"):
        ipaths = df['path'].iloc[indices].values
        itargets = targets[indices]
        ipredictions = predictions[indices]

        assert len(itargets.unique()) == 1
        agg_function = getattr(torch, agg_method)
        itarget = agg_function(itargets)
        iprediction = agg_function(ipredictions)

        ila_results['id'].append(uid)
        ila_results['targets'].append(itarget)
        ila_results['predictions'].append(iprediction)
        ila_results['paths'].append(ipaths)

    ila_results['targets'] = torch.stack(ila_results['targets'])
    ila_predictions = ila_results['predictions']
    ila_predictions = torch.stack(ila_predictions)
    ila_results['predictions'] = torch.cat(
        [(1 - ila_predictions).unsqueeze(1), ila_predictions.unsqueeze(1)], dim=1
    )

    return ila_results
