"""
Generic helpers for file/string operations.
"""
import re


class DotDict(dict):
    """Allows dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_cough_type(cough_path: str) -> str:
    """Returns cough type {`cough_1`, `cough_2`, `cough_3`} from audio path
	Note: this function is specific to wiai-facility/ dataset.
    """

    # defines regex pattern: example - cough_[integer in 0-9].wav
    pattern = r'cough_[\d]\.wav'
    # finds all matches of the above pattern in cough_path
    # assuming there is only 1 occurence of cough_*.wav
    find = re.findall(pattern, cough_path)[0]
    cough_type = find.split('.')[0]

    return cough_type


def get_audio_type(audio_path: str) -> str:
    """Returns audio type (cough_x, speech, breathing)
	Note: this function is specific to wiai-facility/ dataset.
    """
    if 'cough' in audio_path:
        return get_cough_type(audio_path)
    else:
        if 'audio_1_to_10' in audio_path or 'speech' in audio_path:
            return 'audio_1_to_10'
        if 'breathing' in audio_path:
            return 'breathing'


def get_unique_id(audio_path: str) -> str:
    """Returns unique ID for patient whose audio file is inpu
	Note: this function is specific to wiai-facility/ dataset.
    t"""
    prefix = 'user' if 'user' in audio_path else 'patient'
    # define regex pattern: example - {user/patient}_[alphabet OR numbers]_
    pattern = rf'{prefix}_[a-zA-Z\d]+_'

    return re.findall(pattern, audio_path)[0][:-1]
