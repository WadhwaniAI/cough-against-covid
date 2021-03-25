"""Tests cac.data.audio.AudioItem"""
import unittest
import torch
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from cac.utils.loss import LabelSmoothingLoss


class LabelSmoothingLossTestCase(unittest.TestCase):
    """Class to run tests on LabelSmoothingLoss"""
    def test_equal_to_crossentropy_smoothing_zero(self):
        """Tests equality to cross-entropy loss with zero smoothing"""
        y_true = torch.tensor([1, 1, 0, 1, 0])
        y_pred = torch.tensor([
            [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
        ])
        smoothing_loss = LabelSmoothingLoss(
            max_smoothness=0, num_classes=2,
            deterministic=True)(y_pred, y_true)

        ce_loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        self.assertEqual(smoothing_loss, ce_loss)

    def test_unequal_to_crossentropy_smoothing_nonzero(self):
        """Tests equality to cross-entropy loss with non-zero smoothing"""
        y_true = torch.tensor([1, 1, 0, 1, 0])
        y_pred = torch.tensor([
            [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
        ])
        smoothing_loss = LabelSmoothingLoss(
            max_smoothness=0.2, num_classes=2,
            deterministic=True)(y_pred, y_true)

        ce_loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        self.assertNotEqual(smoothing_loss, ce_loss)

    def test_unequal_to_crossentropy_smoothing_random(self):
        """Tests equality to cross-entropy loss with random smoothing"""
        y_true = torch.tensor([1, 1, 0, 1, 0])
        y_pred = torch.tensor([
            [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
        ])
        smoothing_loss = LabelSmoothingLoss(
            max_smoothness=0.2, num_classes=2)(y_pred, y_true)

        ce_loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        self.assertNotEqual(smoothing_loss, ce_loss)

    def test_randomness(self):
        """Tests equality for two different random smoothing calls"""
        y_true = torch.tensor([1, 1, 0, 1, 0])
        y_pred = torch.tensor([
            [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
        ])
        criterion = LabelSmoothingLoss(
            max_smoothness=0.2, num_classes=2)
        torch.manual_seed(0)
        smoothing_loss_1, cache_1 = criterion(y_pred, y_true, True)
        smoothing_loss_2, cache_2 = criterion(y_pred, y_true, True)
        self.assertNotEqual(smoothing_loss_1, smoothing_loss_2)
        self.assertNotEqual(cache_1['confidence'], cache_2['confidence'])
        self.assertNotEqual(cache_1['confidence'], cache_2['confidence'])

    def test_randomness_limit(self):
        """Tests random sampling smoothness"""
        y_true = torch.tensor([1, 1, 0, 1, 0])
        y_pred = torch.tensor([
            [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
        ])
        criterion = LabelSmoothingLoss(
            max_smoothness=0.2, num_classes=2)
        torch.manual_seed(0)
        smoothing_loss, cache = criterion(y_pred, y_true, True)
        self.assertTrue(0 <= 1 - cache['confidence'] <= 0.2)

    def test_smooth_labels(self):
        """Tests label smoothing on cuda tensors"""
        y_true = torch.tensor([1, 1, 0, 1, 0])
        y_pred = torch.tensor([
            [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
        ])
        criterion = LabelSmoothingLoss(
            max_smoothness=0.2, num_classes=2, deterministic=True)
        torch.manual_seed(0)
        smoothing_loss, cache = criterion(y_pred, y_true, True)
        correct_target = np.array([
            [0.2, 0.8], [0.2, 0.8], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2]
        ])
        assert_array_almost_equal(cache['smooth_target'], correct_target)

    def test_cuda(self):
        """Tests label smoothing on cuda tensors"""
        if torch.cuda.is_available():
            y_true = torch.tensor([1, 1, 0, 1, 0]).cuda()
            y_pred = torch.tensor([
                [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
            ]).cuda()
            criterion = LabelSmoothingLoss(
                max_smoothness=0.2, num_classes=2)
            torch.manual_seed(0)
            smoothing_loss, cache = criterion(y_pred, y_true, True)
            self.assertTrue(0 <= 1 - cache['confidence'] <= 0.2)

    def test_smoothing_no_zero_targets(self):
        """Tests that smoothing returns no zero label in the target"""
        y_true = torch.tensor([1, 1, 0, 1, 0])
        y_pred = torch.tensor([
            [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
        ])
        criterion = LabelSmoothingLoss(
            max_smoothness=0.2, num_classes=2)
        torch.manual_seed(0)
        _, cache = criterion(y_pred, y_true, True)
        self.assertEqual(
            len(torch.nonzero(cache['smooth_target'], as_tuple=False)), 10)

    def test_reduction_none(self):
        """Tests smoothing with reduction='none'"""
        y_true = torch.tensor([1, 1, 0, 1, 0])
        y_pred = torch.tensor([
            [3, 3.5], [2, 2.9], [5, 5.2], [3.2, 2.4], [3.3, 2.3]
        ])
        criterion = LabelSmoothingLoss(
            max_smoothness=0.2, num_classes=2, reduction='none')
        torch.manual_seed(0)
        smoothing_loss, cache = criterion(y_pred, y_true, True)
        self.assertEqual(smoothing_loss.shape, (5,))


if __name__ == "__main__":
    unittest.main()
