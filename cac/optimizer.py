"""Defines Factory object to register various optimizers"""
from typing import Any
import math
from abc import ABC, abstractmethod
from cac.factory import Factory
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR, \
    OneCycleLR, MultiStepLR

optimizer_factory = Factory()
optimizer_factory.register_builder('SGD', SGD)
optimizer_factory.register_builder('Adam', Adam)
optimizer_factory.register_builder('AdamW', AdamW)


class Scheduler(ABC):
    """Base class for custom scheduler to inherit from"""
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1
        return self.get_value()

    @abstractmethod
    def get_value(self):
        """Get updated value for the current step"""
        raise NotImplementedError


class Polynomial(Scheduler):
    """Scheduler with polynomial relation with time

    :param power: exponent to be applied to the time steps
    :type power: float
    """
    def __init__(self, power: float):
        super(Polynomial, self).__init__()
        self.power = power

    def get_value(self):
        return math.pow(self.step_count, self.power)


scheduler_factory = Factory()
scheduler_factory.register_builder('ReduceLROnPlateau', ReduceLROnPlateau)
scheduler_factory.register_builder('StepLR', StepLR)
scheduler_factory.register_builder('MultiStepLR', MultiStepLR)
scheduler_factory.register_builder('CyclicLR', CyclicLR)
scheduler_factory.register_builder('1cycle', OneCycleLR)
scheduler_factory.register_builder('Polynomial', Polynomial)
