"""Defines Factory"""
from typing import Any


class Factory:
    """Factory class to build new objects"""

    def __init__(self):
        self._builders = dict()

    def register_builder(self, key: str, builder: Any):
        """Registers a new builder into the factory
        Args:
            key (str): key corresponding to the builder
            builder (Any): Builder object
        """
        self._builders[key] = builder

    def create(self, key: str, **kwargs):
        """Instantiates a new builder object, once it's registered
        Args:
            key (str): key corresponding to the builder
            **kwargs: keyword arguments
        Returns:
            Any: Returns an object corresponding to the dataset builder
        Raises:
            ValueError: If dataset builder is not registered, raises an exception
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)
