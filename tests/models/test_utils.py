"""Tests cac.models.utils"""
import unittest
import math
import numpy as np
from scipy.special import softmax
import torch
from cac.models.utils import logit
from numpy.testing import assert_array_almost_equal


class ModelsUtilsTestCase(unittest.TestCase):
    """Class to run tests on cac.models.utils
    """

    def test_logit(self):
        """Tests `logit` function"""
        x = 0.6
        sigmoid_x = 1 / (1 + math.exp(-x))
        x_ = logit(sigmoid_x)
        assert_array_almost_equal(x, x_)

    def test_logit_extreme_cases(self):
        """Tests `logit` function for extreme cases"""
        z = 0.0
        x = logit(z)
        self.assertTrue(np.abs(x) != np.inf)

        z = 1.0
        x = logit(z)
        self.assertTrue(np.abs(x) != np.inf)

    def test_logit_for_softmax_inverse(self):
        """Tests logit function used for 2D softmax inverse"""

        # original logit output
        x = np.array([-1.0, 0.88])

        # softmax conversion
        softmax_x = softmax(x)

        # use logit to get inverse of softmax
        a = softmax_x[1]
        x_ = np.array([0.0, logit(a)])

        softmax_x_ = softmax(x_)

        assert_array_almost_equal(softmax_x_, softmax_x)
    
if __name__ == "__main__":
    unittest.main()
