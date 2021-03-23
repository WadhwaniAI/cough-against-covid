import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class TransformDict(TypedDict):
    """
    Defines novel type for type-hinting transform dict

    Each transformation is defined by a dict is of the form: {
        'name': <name of the transform>,
        'params': dict of named parameters for this transform
    }
    """
    name: str
    params: dict


class DatasetConfigDict(TypedDict):
    """
    Defines novel type for type-hinting dataset config dict

    Each dataset is defined by a dict of the form: {
        'name': <name of the dataset>,
        'version': version of the dataset to be used
        'mode': train/test/val split of the dataset
    }
    """
    name: str
    version: str
    mode: str


class LayerConfigDict(TypedDict):
    """
    Defines novel type for type-hinting layer config dict

    Each layer is defined by a dict of the form: {
        'name': args
    }

    where `name` is the name of the class corresponding to the layer and
    `args` is a dictionary containing values for different arguments used to
    define the layer.
    """
    name: dict
