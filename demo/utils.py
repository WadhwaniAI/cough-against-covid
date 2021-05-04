"""
Utility functions required for demo (forward passing single input wav file to pre-trained models)
"""
import librosa
import numpy as np
import torch
from cac.data.audio import AudioItem
from cac.data.transforms import DataProcessor
import pandas as pd

# SOME CONSTANTS
CONTEXTUAL_DATA_STATS = {'enroll_patient_age': {'mean': 36.97, 'std': 13.50},
                         'enroll_patient_temperature': {'mean': 97.23, 'std': 1.24},
                         'enroll_days_with_cough': {'mean': 1.22, 'std': 3.14},
                         'enroll_days_with_shortness_of_breath': {'mean': 0.39, 'std': 2.05},
                         'enroll_days_with_fever': {'mean': 1.19, 'std': 3.44}}

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

def _preprocess_raw_context_data(input_):
    continuous_var = ['enroll_patient_age', 'enroll_patient_temperature', 'enroll_days_with_cough',
                      'enroll_days_with_shortness_of_breath', 'enroll_days_with_fever']
    for var in continuous_var:
        input_[var] = (input_[var] - CONTEXTUAL_DATA_STATS[var]['mean']) / CONTEXTUAL_DATA_STATS[var]['std']
    return input_

def check_data_correct_format(input_):
    assert (input_['enroll_patient_age'] > 1 and input_['enroll_patient_age'] < 100), "Age incorrectly Added, Needs to between 1 and 100"
    assert (input_['enroll_patient_temperature'] >= 95 and input_['enroll_patient_temperature'] <= 104), "Temperature incorrectly Added, needs to between 95 and 104"
    for var in ['cough', 'fever', 'shortness_of_breath']:
        assert input_[f'enroll_days_with_{var}'] >= 0, f"Days with Symptom {var} cannot be negative"
        if input_[f'enroll_{var}'] == 0.0:
            assert input_[f'enroll_days_with_{var}'] == 0, f"Patient Needs to have the symptom {var} for 'enroll_days_with_{var}' to be >0"
    assert input_['enroll_travel_history'] in [0, 1, 2, 3], 'Enroll Travel History has Four Options : {0 : No, 1 : Other Country, 2: Other District, 3: Other State}'
    for var in ['enroll_contact_with_confirmed_covid_case', 'enroll_health_worker', 'enroll_fever', 'enroll_cough', 'enroll_shortness_of_breath']:
        assert input_[var] in [0, 1], f'Input to {var} should be binary [0 : No, 1 : Yes]'