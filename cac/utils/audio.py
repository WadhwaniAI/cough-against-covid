import librosa
import numpy as np
from IPython.display import display, HTML, clear_output, Audio


def load_audio(filepath: str, start: float = None, end: float = None):
    """Loads an audio file.

    :param filepath: path to the audio file
    :type filepath: str
    :param start: start duration to load the audio (in secs)
    :type start: float
    :param end: end duration to load the audio (in secs)
    :type end: float
    :returns: tuple containing signal and sampling rate of audio file
    """
    signal, rate = librosa.load(filepath, sr=44100)
    start_idx = 0 if start is None else int(start * rate)
    end_idx = -1 if end is None else int(end * rate)
    signal = signal[start_idx: end_idx]

    return signal, rate


def listen(signal: np.ndarray, rate: int):
    """Displays audio file to listen in a notebook

    :param signal: input audio file signal
    :type signal: np.ndarray
    :param rate: sampling rate of the audio file
    :type rate: int
    """
    display(Audio(data=signal, rate=rate))


def listen_to_audiofile(filepath: str):
    """Loads and displays audio file to listen in a notebook

    :param filepath: path to the audio file
    :type filepath: str
    """
    signal, rate = load_audio(filepath)
    listen(signal, rate)


def get_duration(filepath):
    """Returns the duration of audio file"""
    return librosa.get_duration(filename=filepath)
