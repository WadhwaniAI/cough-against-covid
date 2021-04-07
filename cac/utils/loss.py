"""Defines custom loss functions"""
from typing import Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cac.factory import Factory


class LabelSmoothingLoss(nn.Module):
    """
    Computes the cross-entropy loss between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w)

    Since the PyTorch version of CrossEntropyLoss only accepts an integer
    as the ground truth, we are using KL-divergence as a proxy for
    cross-entropy as the two differ by a constant which is not relevant
    for training.

    :param max_smoothness: maximum smoothness to apply. For
        `deterministic=True`, this corresponds to the smoothing value.
    :type max_smoothness: float
    :param num_classes: total number of classes
    :type num_classes: int
    :param deterministic: whether to apply label smoothing determinstically
        or stochastically. For `deterministic=True`, `max_smoothness` is used
        as the smoothing value. Else, for each forward pass, the smoothing
        value is uniformly sampled between (min_smoothness, max_smoothness).
    :type deterministic: bool, defaults to False
    :param min_smoothness: minimum smoothness to apply. Only used when
        `deterministic=False`.
    :type min_smoothness: float, defaults to 0.
    :param reduction: Refer to `torch.nn.KLDivLoss` for more details
    :type reduction: str, defaults to 'batchmean'
    """
    def __init__(
            self, max_smoothness: float, num_classes: int,
            deterministic: bool = False, min_smoothness: float = 0.,
            reduction: str = 'batchmean'):
        assert 0.0 <= max_smoothness <= 1.0
        assert 0.0 <= min_smoothness <= 1.0
        self.max_smoothness = max_smoothness
        self.min_smoothness = min_smoothness
        self.num_classes = num_classes
        self.deterministic = deterministic
        self.reduction = reduction
        super(LabelSmoothingLoss, self).__init__()

        # for deterministic label smoothing, max_smoothness is
        # considered as the amount of smoothing to apply
        if deterministic:
            label_smoothing = self.max_smoothness
            # without label smoothing, all classes other than the ground
            # truth class has label 0 and the true class has label 1
            # with label smoothing, the given smoothing value is distributed
            # evenly across all classes other than the true class and the true
            # class is assigned a label of 1 - smoothing. When smoothing = 0.1
            # for a binary classification task, the wrong class is given a
            # label of 0.1 and the true class is assigned 0.9.
            smoothing_value = label_smoothing / (num_classes - 1)
            one_hot = torch.full((num_classes,), smoothing_value)
            self.register_buffer('one_hot', one_hot.unsqueeze(0))
            self.confidence = 1.0 - label_smoothing

    def forward(
            self, output: torch.FloatTensor,
            target: torch.LongTensor,
            return_cache: bool = False) -> Union[torch.FloatTensor,
                                                 Tuple[torch.FloatTensor,
                                                       dict]]:
        """
        :param output: logits of shape = batch_size x n_classes
        :type output: torch.FloatTensor
        :param target: ground truth labels of shape = batch_size
        :type target: torch.LongTensor
        :param return_cache: whether to return intermediate values
        :type return_cache: bool, defaults to False
        """
        # convert logits to log probabilities
        log_probs = F.log_softmax(output, dim=-1)

        # get the smoothing value for the non-deterministic case
        if not self.deterministic:
            # for stochastic label smoothing, randomly sample the
            # amount of smoothing to be applied
            label_smoothing = np.random.uniform(
                self.min_smoothness, self.max_smoothness)

            # set the smoothing value as described for the deterministic
            # case in __init__()
            smoothing_value = label_smoothing / (self.num_classes - 1)
            one_hot = torch.full((self.num_classes,), smoothing_value)
            self.one_hot = one_hot.unsqueeze(0)

            # set the label for the true class
            self.confidence = 1.0 - label_smoothing

        # smoothen the labels
        smooth_target = self.one_hot.repeat(target.size(0), 1).to(
            log_probs.device)
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = F.kl_div(log_probs, smooth_target,
                        reduction=self.reduction)

        if self.reduction == 'none':
            # take the sum across the class dimension to get
            # one value for each instance
            loss = loss.sum(dim=-1)

        if not return_cache:
            return loss

        cache = {
            'smooth_target': smooth_target,
            'log_probs': log_probs,
            'confidence': self.confidence
        }

        return loss, cache


class BCEWithLogitsLoss(nn.Module):
    """
    Computes binary cross entropy loss based on nn.BCEWithLogitsLoss
    but has additional checks on shape and dtype of predictions and targets.
    """
    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(**kwargs)

    def __call__(self, output: torch.float32, target: torch.float32):
        """
        :param output: predicted classification score, typically output of neural network,
            needs to be of shape (B, 1) or (B,) single-label case or (B, L) for multi-label case
        :type output: torch.float32
        :param target: target class, needs to be of shape (B, 1) or (B,) single-label case or
            (B, L) for multi-label case, should be of same type as output
        :type target: torch.float32
        """
        output, target = self._check_inputs(output, target)
        return self.criterion(output, target)

    def _check_inputs(self, output, target):
        # check if output is float32
        if not output.dtype != torch.float32:
            output = output.type(torch.float32)

        # make target of same type as output
        target = target.type_as(output)

        # depending on target.shape, reshape output
        if len(target.shape) == 1:
            # simple single-label case (B,)
            if len(output.shape) not in [1, 2]:
                raise ValueError(
                    f'output shape must be (B, 1) or (B,) if target shape is (B, {target.shape[-1]})'
                )
            if len(output.shape) == 2:
                output = output.squeeze(1)

        elif len(target.shape) == 2:
            if target.shape[-1] == 1:
                # again, single-label case (B, 1)
                if len(output.shape) not in [1, 2]:
                    raise ValueError(
                        f'output shape must be (B, 1) or (B,) if target shape is (B, {target.shape[-1]})'
                    )
                if len(output.shape) == 1:
                    output = output.unsqueeze(1)
            else:
                # multi-label case (B, L)
                assert output.shape == target.shape
        else:
            raise ValueError(
                f'target shape must be (B, l), l in [0, .., L] but found {target.shape}'
            )
        
        return output, target


loss_factory = Factory()
loss_factory.register_builder('cross-entropy', nn.CrossEntropyLoss)
loss_factory.register_builder('binary-cross-entropy', BCEWithLogitsLoss)
loss_factory.register_builder('label-smoothing', LabelSmoothingLoss)
