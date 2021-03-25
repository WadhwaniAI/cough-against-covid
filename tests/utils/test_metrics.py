"""Tests cac.data.audio.AudioItem"""
import unittest
import torch
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.metrics import precision_recall_curve
from cac.utils.metrics import PrecisionAtRecall, ConfusionMatrix, \
    SpecificityAtSensitivity


class PrecisionAtRecallTestCase(unittest.TestCase):
    """Class to run tests on PrecisionAtRecall"""
    def test_recall_specified(self):
        """Checks the case when threshold for recall specified"""
        y_true = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        y_pred = torch.tensor(
            [0.8, 0.7, 0.69, 0.72, 0.68, 0.58, 0.83, 0.65, 0.91, 0.3])
        precision, recall, threshold = PrecisionAtRecall(0.9)(y_true, y_pred)
        self.assertEqual(recall, 1)

    def test_varying_precisions(self):
        """Checks the case when precision changes for varying thresholds"""
        y_true = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        y_pred = torch.tensor(
            [0.8, 0.7, 0.75, 0.72, 0.68, 0.58, 0.83, 0.65, 0.91, 0.71, 0.64, 0.62])
        precision, recall, threshold = PrecisionAtRecall(0.9)(y_true, y_pred)
        self.assertEqual(recall, 0.9)
        self.assertEqual(precision, 1)

    def test_recall_zero(self):
        """Checks the case when recall specified = 0"""
        y_true = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        y_pred = torch.tensor(
            [0.8, 0.7, 0.69, 0.72, 0.68, 0.58, 0.83, 0.65, 0.91, 0.3])
        precision, recall, threshold = PrecisionAtRecall(0)(y_true, y_pred)
        self.assertTrue(recall == 0.0)

    def test_bad_prediction_shape(self):
        """Checks the case when prediction shape is wrong"""
        y_true = torch.zeros(10)
        y_pred = torch.zeros((10, 2))

        with self.assertRaises(ValueError):
            _ = PrecisionAtRecall(0.9)(y_true, y_pred)

    def test_bad_gt_values(self):
        """Checks the case when more than 2 classes are present in ground truth"""
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1, 0])

        with self.assertRaises(ValueError):
            _ = PrecisionAtRecall(0.9)(y_true, y_pred)


class SpecificityAtSensitivityTestCase(unittest.TestCase):
    """Class to run tests on SpecificityAtSensitivity"""
    def test_sensitivity_specified(self):
        """Checks the case when threshold for sensitivity specified"""
        y_true = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        y_pred = torch.tensor(
            [0.8, 0.7, 0.75, 0.72, 0.68, 0.58, 0.83, 0.65, 0.91, 0.71, 0.64, 0.62])
        specificity, sensitivity, threshold = SpecificityAtSensitivity(
            0.9)(y_true, y_pred)
        self.assertEqual(sensitivity, 0.9)
        self.assertEqual(specificity, 1)

    def test_bad_prediction_shape(self):
        """Checks the case when prediction shape is wrong"""
        y_true = torch.zeros(10)
        y_pred = torch.zeros((10, 2))

        with self.assertRaises(ValueError):
            _ = SpecificityAtSensitivity(0.9)(y_true, y_pred)

    def test_bad_gt_values(self):
        """Checks the case when more than 2 classes are present in ground truth"""
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1, 0])

        with self.assertRaises(ValueError):
            _ = SpecificityAtSensitivity(0.9)(y_true, y_pred)


class ConfusionMatrixTestCase(unittest.TestCase):
    """Class to run tests on ConfusionMatrix"""
    @classmethod
    def setUpClass(cls):
        cls.y_true = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        cls.y_pred = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 0, 0])

    def test_confusion_matrix(self):
        """Checks the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        assert_array_equal(cm.cm, [[3, 2], [3, 2]])

    def test_tp(self):
        """Checks TP in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.tp, 2)

    def test_tn(self):
        """Checks TN in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.tn, 3)

    def test_fp(self):
        """Checks FP in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.fp, 2)

    def test_fn(self):
        """Checks FN in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.fn, 3)

    def test_sensitivity(self):
        """Checks sensitivity in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.sensitivity, 0.4)

    def test_specificity(self):
        """Checks specificity in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.specificity, 0.6)

    def test_plr(self):
        """Checks PLR in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.plr, 1.0)

    def test_nlr(self):
        """Checks NLR in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.nlr, 1.0)

    def test_ppv(self):
        """Checks PPV in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.ppv, 0.02)

    def test_npv(self):
        """Checks NPV in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.npv, 0.98)

    def test_overall_accuracy(self):
        """Checks overall accuracy in the usual case"""
        cm = ConfusionMatrix([0, 1])
        cm(self.y_true, self.y_pred)
        self.assertEqual(cm.overall_accuracy, 0.596)

    def test_tp_before_cm(self):
        """Checks calling tp before computing confusion matrix"""
        cm = ConfusionMatrix([0, 1])
        with self.assertRaises(ValueError):
            cm.tp

    def test_tp_multi_class(self):
        """Checks calling tp on multi-class problem"""
        cm = ConfusionMatrix([0, 1, 2])
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1, 2])
        cm(y_true, y_pred)
        with self.assertRaises(ValueError):
            cm.tp


if __name__ == "__main__":
    unittest.main()
