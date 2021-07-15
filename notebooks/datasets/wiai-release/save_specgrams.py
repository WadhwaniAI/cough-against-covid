import os
from cac.utils.io import read_yml
from os.path import join
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

from cac.data.audio import AudioItem
from cac.data.transforms import DataProcessor
from cac.utils.viz import plot_raw_audio_signal, plot_spectrogram_image

DATA_RELEASE_DIR = "/data/wiai-release-audios/"
SPECTROGRAM_DATA_RELEASE_DIR = "/data/wiai-release-spectrograms/"
os.makedirs(DATA_RELEASE_DIR, exist_ok = True)
os.makedirs(SPECTROGRAM_DATA_RELEASE_DIR, exist_ok = True)

dataset_audio_files = glob(join(DATA_RELEASE_DIR, "processed", "audio", "*.wav"))

transforms_cfg = [
    {
        "name": "ToTensor",
        "params": {"device": "cpu"}
    },
    {
        "name": "Resample",
        "params": {
            "orig_freq": 44100,
            "new_freq": 16000
        }
    },
    {
        "name": "Spectrogram",
        "params": {
            "n_fft": 512,
            "win_length": 512,
            "hop_length": 160
        }
    },
    {
        "name": "ToNumpy",
        "params": {}
    },
]

signal_transform = DataProcessor(transforms_cfg)

os.makedirs(os.path.join(SPECTROGRAM_DATA_RELEASE_DIR, 'processed', 'spectrograms'), exist_ok = True)
spec_save_path = os.path.join(SPECTROGRAM_DATA_RELEASE_DIR, 'processed', 'spectrograms')

for audio_file in tqdm(dataset_audio_files):
    audio_file_name = audio_file.split('/')[-1].split('.')[0]
    item = AudioItem(path=audio_file)
    signal = item.load()["signal"]
    transformed_signal = signal_transform(signal)
    
    dest_path = os.path.join(spec_save_path, f'{audio_file_name}.npy')
    np.save(dest_path, transformed_signal)