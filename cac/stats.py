"""Defines Factory object to register various sklearn methods"""
from typing import Any
from cac.factory import Factory
import numpy as np
from scipy.stats import skew, kurtosis

class Mean:
    """docstring for Mean"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return np.mean(x, axis=self.axis)


class Median:
    """docstring for Median"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return np.median(x, axis=self.axis)


class Min:
    """docstring for Min"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return np.min(x, axis=self.axis)


class Max:
    """docstring for Max"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return np.max(x, axis=self.axis)


class RMS:
    """docstring for Max"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return np.sqrt(np.mean(x**2, axis=self.axis))


class FirstQuartile:
    """docstring for Max"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return np.percentile(x, 25, axis=self.axis)


class ThirdQuartile:
    """docstring for Max"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return np.percentile(x, 75, axis=self.axis)


class IQR:
    """docstring for Max"""
    def __init__(self, axis : int = -1):
        self.axis = axis
        self.Q1 = FirstQuartile(axis=axis)
        self.Q3 = ThirdQuartile(axis=axis)

    def __call__(self, x: np.ndarray):
        return (self.Q1(x) - self.Q3(x)) / 2


class StandardDeviation:
    """docstring for Max"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return np.std(x, axis=self.axis)


class Skewness:
    """docstring for Max"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return skew(x, axis=self.axis)


class Kurtosis:
    """docstring for Max"""
    def __init__(self, axis : int = -1):
        self.axis = axis

    def __call__(self, x: np.ndarray):
        return kurtosis(x, axis=self.axis)


DEFAULT_STATS = ['Mean', 'Median', 'Min', 'Max', 'RMS', 'FirstQuartile', \
    'ThirdQuartile', 'IQR', 'StandardDeviation', 'Skewness', 'Kurtosis']

factory = Factory()
factory.register_builder('Mean', Mean)
factory.register_builder('Median', Median)
factory.register_builder('Min', Min)
factory.register_builder('Max', Max)
factory.register_builder('RMS', RMS)
factory.register_builder('Skewness', Skewness)
factory.register_builder('Kurtosis', Kurtosis)
factory.register_builder('FirstQuartile', FirstQuartile)
factory.register_builder('ThirdQuartile', ThirdQuartile)
factory.register_builder('StandardDeviation', StandardDeviation)
factory.register_builder('IQR', IQR)
