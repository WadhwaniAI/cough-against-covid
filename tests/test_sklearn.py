"""Tests cac.sklearn"""
import unittest
from sklearn.svm import SVC
from cac.sklearn import factory as sklearn_factory


class SklearnTestCase(unittest.TestCase):
    """Class to run tests on sklearn objects"""
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_sklearn_factory(self):
        # this will be part of a config file
        config = {
            'name': 'SVM',
            'params': {
                'C': 0.1
            }
        }

        svm = sklearn_factory.create(
            config['name'], **config['params']
        )

        self.assertTrue(isinstance(svm, SVC))
        self.assertEqual(svm.C, 0.1)


if __name__ == "__main__":
    unittest.main()
