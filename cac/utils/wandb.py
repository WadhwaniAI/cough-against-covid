import random
from typing import List, Any
import numpy as np
import torch
import wandb
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

from cac.utils.viz import fig2im
from cac.decomposition.methods import factory as decomposition_factory

# set precision
torch.set_printoptions(precision=4)


def get_confusion_matrix(
        matrix_array: np.ndarray, classes: List[str]
        ) -> np.ndarray:
    """Return image corresponding to confusion matrix

    :param matrix_array: N x N confusion matrix, where N is the number of classes
    :type matrix_array: np.ndarray
    :param classes: list of the class names
    :type classes: List[str]
    :returns: numpy array corresponding to the figure where the confusion matrix has been plotted
    """

    ticklabels = {}

    for k in ('x', 'y'):
        key = k + 'ticklabels'
        ticklabels[key] = [''] + classes + ['']

    tickpositions = {}
    for k in ('x', 'y'):
        key = k + 'ticks'
        tickpositions[key] = [0] + (np.array(list(range(0, len(classes)))) \
            + 0.5).tolist() + [len(classes)]

    fmt = 'd'

    ax = sns.heatmap(data=matrix_array,
                     fmt=fmt,
                     cbar=False,
                     annot=True,
                     square=True,
                     cmap=ListedColormap(['#F5F5F5']),
                     linecolor='#DCDCDC',
                     linewidths=1,
                     xticklabels=[],
                     yticklabels=[])

    ax.set_xticks(tickpositions['xticks'])
    ax.set_yticks(tickpositions['yticks'])
    ax.set_xticklabels(ticklabels['xticklabels'])
    ax.set_yticklabels(ticklabels['yticklabels'])

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_tick_params(rotation=90)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_verticalalignment('center')

    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    return fig2im(ax.get_figure())


def get_indices(targets: torch.Tensor, per_class: bool = True,
                count: int = 4, seed: int = 0) -> List[int]:
    """Samples indices to visualize

    :param targets: ground truths
    :type targets: torch.Tensor
    :param per_class: whether count means count per class, defaults to True
    :type per_class: bool
    :param count: number of audios to return, defaults to 4
    :type count: int, optional
    :param seed: random seed, defaults to 0
    :type seed: int, optional
    """
    if not per_class:
        indices = list(range(len(targets)))
        indices = random.Random(seed).sample(indices, count)
    else:
        indices = []
        unique_targets, inverse_indices = torch.unique(
            targets, return_inverse=True)
        for index in range(len(unique_targets)):
            target_indices = torch.nonzero(
                inverse_indices == index, as_tuple=True)
            target_indices = target_indices[0].tolist()
            _indices = random.Random(seed).sample(
                target_indices, min(count, len(target_indices)))
            indices.extend(_indices)

    return indices


def get_audios(
        items: List[str], predictions: List[Any], targets: List[Any]
        ) -> List[wandb.Audio]:
    """Returns a list of wandb.Audio objects

    :param items: list of items of all inputs
    :type items: List[str]
    :param predictions: model predictions
    :type predictions: List[Any]
    :param targets: ground truths
    :type targets: List[Any]
    """
    audios = []

    for item, prediction, target in zip(items, predictions, targets):

        audio = item.load()
        caption = 'Pred: {} GT: {}'.format(prediction, target)

        audios.append(
            wandb.Audio(
                audio['signal'], caption=caption, sample_rate=audio['rate']))

    return audios


def get_images(
        inputs: List[Any], predictions: List[Any], targets: List[Any]
        ) -> List[wandb.Image]:
    """Returns a list of wandb.Audio objects

    :param paths: list of paths of all inputs
    :type paths: List[str]
    :param predictions: model predictions
    :type predictions: List[Any]
    :param targets: ground truths
    :type targets: List[Any]
    """
    images = []

    for _input, prediction, target in zip(inputs, predictions, targets):

        caption = 'Pred: {} GT: {}'.format(prediction, target)

        # import ipdb; ipdb.set_trace()
        images.append(
            wandb.Image(torch.Tensor(_input), caption=caption))

    return images


def get_decomposition_embeddings(
        features: List[Any], method: str = 'TSNE') -> torch.Tensor:
    """Projects the input to a lower dimensional embedding

    :param features: features to visualize
    :type features: List[str]
    :param method: projection method, defaults to 'TSNE'
    :type method: str
    """
    params = {'n_components': 2}
    method = decomposition_factory.create(method, **params)
    embeddings = method.fit_transform(features)
    return embeddings


def plot2d(
        X: [torch.Tensor, np.ndarray], targets: List[Any],
        classes: List[str],
        ) -> List[wandb.Image]:
    """Returns a list of wandb.Image objects

    :param X: 2D array to visualize
    :type X: List[str]
    :param targets: ground truth integers used to color each point
    :type targets: List[Any]
    :param classes: list of classes representing ground truth
    :type classes: List[str]
    """
    targets = np.array([classes[target] for target in targets])
    unique_targets = np.unique(targets)
    colors = cm.plasma(np.linspace(0, 1, len(unique_targets)))

    f, ax = plt.subplots(1, figsize=(10, 10))

    for (i, target), color in zip(enumerate(unique_targets), colors):
        indices = np.where(targets == target)
        num = len(indices[0])
        ax.scatter(
            X[indices, 0], X[indices, 1],
            label='{} : {}'.format(target, num), color=color)

    ax.set_ylabel('Component 2')
    ax.set_xlabel('Component 1')
    ax.grid()
    ax.legend(loc='best')
    image = wandb.Image(plt)
    plt.close()
    return image
