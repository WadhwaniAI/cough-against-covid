from typing import Tuple
import logging
from os.path import join
import sys
from cac.config import DATA_ROOT
from cac.utils.logger import color
from cac.utils.io import read_yml
from cac.utils.typing import DatasetConfigDict


def read_dataset_from_config(dataset_config: DatasetConfigDict) -> dict:
    """
    Loads and returns the dataset version file corresponding to the
    dataset config.

    :param dataset_config: dict containing (name, version, mode) corresponding to a dataset.
    :type dataset_config: DatasetConfigDict
    :returns: dict of list of filepaths, list of labels etc contained in the version file
    """
    name, version, mode = dataset_config['name'], dataset_config['version'], \
        dataset_config['mode']

    version_fpath = join(
        DATA_ROOT, name, 'processed/versions', version + '.yml')

    logging.info(color("=> Loading dataset version file: [{}, {}, {}]".format(
        name, version, mode)))

    version_file = read_yml(version_fpath)

    return version_file[mode]
