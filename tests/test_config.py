"""Tests cac.config.Config"""
import unittest
from cac.utils.io import read_yml
from cac.config import Config


class ConfigTestCase(unittest.TestCase):
    """Class to run tests on Config"""
    @classmethod
    def setUpClass(cls):
        pass

    def test_default_config(self):
        """Test creating Config object with default.yml"""
        version = 'default.yml'
        cfg = Config(version)

        self.assertIn('data', dir(cfg))
        self.assertIn('network', dir(cfg))


if __name__ == "__main__":
    unittest.main()
