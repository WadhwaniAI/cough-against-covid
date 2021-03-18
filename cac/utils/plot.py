"""Utility functions for plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical_attribute(
        df, attribute, hue=None, title=None, ax=None, figsize=(8, 6), show=False, kwargs=dict()
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.countplot(data=df, x=attribute, ax=ax, hue=hue, **kwargs)
    ax.grid()
    if title is None:
        title = f'Countplot of {attribute}'
    ax.set_title(title)
    ax.set_xlabel('')

    patches = ax.patches
    for patch in patches:
        x, _ = patch.xy
        counts = patch.get_height()
        ax.text(x + 0.1, counts + 25, counts)

    if show:
        plt.show()

