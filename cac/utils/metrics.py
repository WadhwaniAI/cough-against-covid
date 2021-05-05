"""Defines custom metrics"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve
from cac.factory import Factory


class Metric(ABC):
    """Base class to be inherited by all metric classes"""
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class ConfusionMatrix(Metric):
    """Computes the confusion matrix

    Additionally, for binary classification, returns the true positives,
    true negatives, false positives, false_negatives.

    :param classes: list of classes
    :type classes: List[Any]
    :param prevalence: prevalence in the case of disease
    :type prevalence: float, defaults to 0.02.
    """
    def __init__(self, classes: List[Any], prevalence=0.02):
        self.classes = classes
        self.prevalence = prevalence

    def __call__(
            self, y_true: np.ndarray, y_pred: np.ndarray
            ) -> np.ndarray:
        self.cm = confusion_matrix(
            y_true, y_pred, np.arange(len(self.classes)))

    @property
    def tp(self):
        """Returns the number of true positives"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('true positives are only valid for 2 classes')
        return self.cm[1, 1]

    @property
    def tn(self):
        """Returns the number of true negatives"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('true negatives are only valid for 2 classes')
        return self.cm[0, 0]

    @property
    def fp(self):
        """Returns the number of false positives"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('false positives are only valid for 2 classes')
        return self.cm[0, 1]

    @property
    def fn(self):
        """Returns the number of false negatives"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('false negatives are only valid for 2 classes')
        return self.cm[1, 0]

    @property
    def sensitivity(self):
        """Returns the sensitivity"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('sensitivity is only valid for 2 classes')
        epsilon = np.finfo(float).eps
        sensitivity = (self.tp + epsilon) / (self.tp + self.fn + epsilon)
        return np.round(sensitivity, 4)

    @property
    def specificity(self):
        """Returns the specificity"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('specificity is only valid for 2 classes')
        epsilon = np.finfo(float).eps
        specificity = (self.tn + epsilon) / (self.tn + self.fp + epsilon)
        return np.round(specificity, 4)

    @property
    def overall_accuracy(self):
        """Returns the overall_accuracy"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('overall_accuracy is only valid for 2 classes')
        epsilon = np.finfo(float).eps
        overall_accuracy = self.sensitivity * self.prevalence + \
            self.specificity * (1 - self.prevalence)
        return np.round(overall_accuracy, 4)

    @property
    def ppv(self):
        """Returns the positive predictive value"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('PPV is only valid for 2 classes')
        epsilon = np.finfo(float).eps
        numerator = self.sensitivity * self.prevalence
        denominator = self.sensitivity * self.prevalence + \
            (1 - self.specificity) * (1 - self.prevalence)
        ppv = (numerator + epsilon) / (denominator + epsilon)
        return np.round(ppv, 4)

    @property
    def npv(self):
        """Returns the negative predictive value"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('NPV is only valid for 2 classes')
        epsilon = np.finfo(float).eps
        numerator = self.specificity * (1 - self.prevalence)
        denominator = (1 - self.sensitivity) * self.prevalence + \
            self.specificity * (1 - self.prevalence)
        npv = (numerator + epsilon) / (denominator + epsilon)
        return np.round(npv, 4)

    @property
    def plr(self):
        """Returns the positive likelihood ratio"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('PLR is only valid for 2 classes')
        epsilon = np.finfo(float).eps
        numerator = self.sensitivity
        denominator = 1 - self.specificity
        plr = (numerator + epsilon) / (denominator + epsilon)
        return np.round(plr, 4)

    @property
    def nlr(self):
        """Returns the negative likelihood ratio"""
        if not hasattr(self, 'cm'):
            raise ValueError('confusion matrix has not been computed yet.')
        if len(self.classes) != 2:
            raise ValueError('NLR is only valid for 2 classes')
        epsilon = np.finfo(float).eps
        numerator = 1 - self.sensitivity
        denominator = self.specificity
        nlr = (numerator + epsilon) / (denominator + epsilon)
        return np.round(nlr, 4)


class PrecisionAtRecall(Metric):
    """Computes the maximum precision given the specified recall

    :param recall: desired minimum recall value at which the best precision is
        to be measured
    :type recall: float
    """
    def __init__(self, recall: float):
        self.recall = recall

    def __call__(
            self, y_true: np.ndarray, y_pred: np.ndarray
            ) -> Tuple[float, float, float]:
        self._check_inputs(y_pred, y_true)

        # no positives in the ground truth
        if not len(torch.nonzero(y_true, as_tuple=False)):
            return -1, -1, 0.5

        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_pred)

        if not self.recall:
            # handle the case when required recall is 0
            thresholds = np.append(thresholds, 1)
            precision, recall, threshold = 1, 0, 1
        else:
            # ignore last element as it always has precision = 1 and recall = 0
            precision, recall, threshold = self._get_best_values(
                precisions[:-1], recalls[:-1], thresholds)

        return precision, recall, threshold

    def _get_best_values(
            self, precisions: np.ndarray, recalls: np.ndarray,
            thresholds: np.ndarray) -> int:
        """Returns the values at the best threshold"""
        indices = recalls >= self.recall
        precisions, recalls, thresholds = precisions[indices], \
            recalls[indices], thresholds[indices]
        idx = np.argmax(precisions)
        return precisions[idx], recalls[idx], thresholds[idx]

    def _check_inputs(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Validates the input"""
        if len(y_pred.shape) != 1 or torch.max(y_pred) > 1 or torch.min(y_pred) < 0:
            raise ValueError(
                'PrecisionAtRecall requires prediction probability as input')

        if len(torch.unique(y_true)) > 2:
            raise ValueError(
                'PrecisionAtRecall only supports binary classification tasks')


class SpecificityAtSensitivity(Metric):
    """Computes the maximum specificity given the specified sensitivity

    :param recall: desired minimum sensitivity value at which the best
        specificity is to be measured
    :type recall: float
    """
    def __init__(self, recall: float):
        self.sensitivity = recall

    def __call__(
            self, y_true: np.ndarray, y_pred: np.ndarray
            ) -> Tuple[float, float, float]:
        self._check_inputs(y_pred, y_true)

        # only one label in the ground truth
        if len(torch.unique(y_true)) == 1:
            return -1, -1, 0.5

        fprs, sensitivities, thresholds = roc_curve(y_true, y_pred)
        thresholds = np.clip(thresholds, 0, 1)
        specificities = np.array([1 - fpr for fpr in fprs])

        # get the best index for given sensitivity value
        specificity, sensitivity, threshold = self._get_best_values(
            specificities, sensitivities, thresholds)
        return specificity, sensitivity, threshold

    def _get_best_values(
            self, specificities: np.ndarray, sensitivities: np.ndarray,
            thresholds: np.ndarray) -> int:
        """Returns the index for the best threshold to use"""
        indices = sensitivities >= self.sensitivity
        specificities, sensitivities, thresholds = specificities[indices], \
            sensitivities[indices], thresholds[indices]
        idx = np.argmax(specificities)
        return specificities[idx], sensitivities[idx], thresholds[idx]

    def _check_inputs(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Validates the input"""
        if len(y_pred.shape) != 1 or torch.max(y_pred) > 1 or torch.min(y_pred) < 0:
            raise ValueError(
                'SpecificityAtSensitivity requires prediction probability as input')

        if len(torch.unique(y_true)) > 2:
            raise ValueError(
                'SpecificityAtSensitivity only supports binary classification tasks')


factory = Factory()
factory.register_builder('precision', PrecisionAtRecall)
factory.register_builder('specificity', SpecificityAtSensitivity)
