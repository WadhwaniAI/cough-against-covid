from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt


def fig2data(fig: plt.Figure):
    """Convert a Matplotlib figure to a numpy array with RGBA channels

    :param fig: a matplotlib figure
    :type fig: plt.Figure
    :return: a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the
    # ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig2im(fig: plt.Figure):
    """Convert figure to ndarray

    :param fig: a matplotlib figure
    :type fig: plt.Figure
    """
    img_data = fig2data(fig).astype(np.int32)
    plt.close()
    return img_data[:, :, :3] / 255.


def plot_raw_audio_signal(signal: np.ndarray, title: str ='Raw audio signal'):
    """Plots raw waveform (audio signal)

    :param signal: audio file signal
    :type signal: np.ndarray
    :param title: title of the plot
    :type title: str
    """
    plt.figure(figsize=(23, 4))
    plt.grid()
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    plt.close()


def plot_lr_vs_loss(lrs: List[float], losses: List[float], as_figure: bool = True):
    """Plot learning rate vs loss values during LR range test

    :param lrs: list-like, list of learning rate values
    :param lrs: List[float]
    :param losses: list-like, list of loss values
    :param losses: List[float]
    :param as_plot: whether to return plt.Figure; if False,
        returns numpy image
    :param as_plot: bool, defaults to True
    """
    plt.plot(lrs, losses)
    plt.title('Losses vs learning rate over the epochs')
    figure = plt.gcf()

    if as_figure:
        return figure

    vis_image = fig2im(figure)
    plt.close()
    return vis_image


def plot_classification_metric_curve(
        x: List[Any], y: List[Any], xlabel: str, ylabel: str,
        title: str = None, as_figure: bool = True):
    """Create standard classification metric curves - ROC, PR, etc.

    :param x: list-like, list of values for the x-axis
    :param x: List[Any]
    :param y: list-like, list of values for the y-axis
    :param y: List[Any]
    :param xlabel: list-like, title for the x-axis
    :param xlabel: str
    :param ylabel: list-like, title for the y-axis
    :param ylabel: str
    :param title: list-like, title for the plot
    :param title: str, defaults to None
    :param as_plot: whether to return plt.Figure; if False,
        returns numpy image
    :param as_plot: bool, defaults to True
    """
    figure, ax = plt.subplots(1, figsize=(10, 8))
    ax.grid()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')

    if title is not None:
        plt.title(title)

    if as_figure:
        return figure

    vis_image = fig2im(figure)
    plt.close()
    return vis_image
