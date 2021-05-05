"""Tests cac.factory.Factory"""
import unittest
from cac.factory import Factory


class Dummy:
    """Sample dummy class."""
    def __init__(self, x, y):
        super(Dummy, self).__init__()
        self.x = x
        self.y = y
        self.sum = self.add(x, y)

    @staticmethod
    def add(a, b):
        return a + b


class FactoryTestCase(unittest.TestCase):
    """Class to run tests on Factory"""
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_factory_create_object(self):
        # define the factory
        factory = Factory()

        # first register the builder for the object you want to use
        factory.register_builder('Dummy', Dummy)

        # this will be part of a config file
        dummy_config = {
            'name': 'Dummy',
            'params': {
                'x': 0.1,
                'y': 0.2
            }
        }

        # creates Dummy(x=x, y=y) object based on dummy_config
        dummy = factory.create(
            dummy_config['name'], **dummy_config['params']
        )

        self.assertEqual(dummy.x, 0.1)
        self.assertEqual(dummy.y, 0.2)
        self.assertAlmostEqual(dummy.sum, 0.3)


if __name__ == "__main__":
    unittest.main()
