"""
Utility functions required for demo (forward passing single input wav file to pre-trained models)
"""
import librosa
import numpy as np
import torch

from cac.data.audio import AudioItem
from cac.data.transforms import DataProcessor


def _create_frames(path, frame_length=2000, hop_length=500, min_length=100, label=None, start=None, end=None):
    """
    Function borrowed from cac.data.base.BaseDataset
    to create sliding window frames for a given audio.
    """
    if start is None:
        start = 0.0
    if end is None:
        end = librosa.get_duration(filename=path)
    
    frame_length /= 1000
    hop_length /= 1000
    min_length /= 1000

    # if the file is smaller than frame_length, simply return one audio item
    if end - start < frame_length:
        return [AudioItem(path=path, label=label, start=start, end=end)]

    steps = np.arange(start, end, hop_length)
    items = []
    for step in steps:
        # this indicates the last frame
        if end - step < frame_length:
            # check if it is long enough to be considered
            if end - (step + hop_length) > min_length:
                _start = end - frame_length
                items.append(AudioItem(path=path, label=label, start=_start, end=end))
            break

        _end = step + frame_length
        items.append(AudioItem(path=path, label=label, start=step, end=_end))

    return items


def _process_raw_audio_file(path, signal_transforms, frame_length, hop_length, min_length=100):
    items = _create_frames(
        path=path, frame_length=frame_length, hop_length=hop_length, min_length=min_length
    )
    signal_transformer = DataProcessor(signal_transforms)

    batch = []
    for item in items:
        audio = item.load(as_tensor=True)
        signal = audio['signal']
        signal = signal_transformer(signal)
        batch.append(signal.unsqueeze(dim=0))

    batch = torch.cat(batch)

    return batch