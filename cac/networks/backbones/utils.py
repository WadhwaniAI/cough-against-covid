import logging
from collections import OrderedDict
import torch


def _correct_state_dict(
        loaded_state_dict: OrderedDict,
        model_state_dict: OrderedDict) -> OrderedDict:
    """Only retains key from the `loaded_state_dict` that match with `model_state_dict`

    :param loaded_state_dict: state_dict to be loaded
    :type loaded_state_dict: OrderedDict
    :param model_state_dict: state_dict of the model
    :type model_state_dict: OrderedDict
    :returns: OrderedDict, state_dict compatible for loading with the model
    """
    corrected_state_dict = OrderedDict()
    for key, value in loaded_state_dict.items():
        if key not in model_state_dict or value.shape != model_state_dict[key].shape:
            logging.info(f'Removing {key} from state_dict')
            continue

        corrected_state_dict[key] = value
    return corrected_state_dict
