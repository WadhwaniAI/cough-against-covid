from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib import rc
import colorsys


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_raw_audio_signal_with_markings(signal: np.ndarray, markings: list,
    title: str = 'Raw audio signal with markings'):

    plt.figure(figsize=(23, 4))
    plt.grid()

    plt.plot(signal)
    for value in markings:
        plt.axvline(x=value, c='red')
    plt.xlabel('Time')
    plt.title(title)

    plt.show()
    plt.close()


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
