"""Tests cac.data.classification.ClassificationAnnotationTransform"""
import unittest
from cac.data.transforms import ClassificationAnnotationTransform


class ClassificationAnnotationTransformTestCase(unittest.TestCase):
    """Class to run tests on ClassificationAnnotationTransform"""
    @classmethod
    def setUpClass(cls):
        binary_classes = ['cough']
        multi_classes = ['sneeze', 'cough', 'bark']
        cls.binary_transform = ClassificationAnnotationTransform(binary_classes)
        cls.multi_transform = ClassificationAnnotationTransform(multi_classes)
        cls.binary_transform_no_auto_increment = ClassificationAnnotationTransform(
            binary_classes, auto_increment=False)

    def test_empty_target(self):
        """Checks the case when input target is empty"""
        target = []
        transform_target = self.binary_transform(target)
        self.assertEqual(transform_target, 0)

    def test_no_intersection_binary(self):
        """Checks the case when input target has no intersection with binary_transform"""
        target = ['a', 'b']
        transform_target = self.binary_transform(target)
        self.assertEqual(transform_target, 0)

    def test_no_intersection_multi(self):
        """Checks the case when input target has no intersection with multi_transform"""
        target = ['a', 'b']
        transform_target = self.multi_transform(target)
        self.assertEqual(transform_target, 0)

    def test_one_intersection_binary(self):
        """Checks the case when input target has one intersection with binary_transform"""
        target = ['cough']
        transform_target = self.binary_transform(target)
        self.assertEqual(transform_target, 1)

    def test_one_intersection_multi(self):
        """Checks the case when input target has one intersection with multi_transform"""
        target = ['cough', 'car']
        transform_target = self.multi_transform(target)
        self.assertEqual(transform_target, 2)

    def test_multiple_intersection_multi(self):
        """Checks the case when input target has multiple intersections with multi_transform"""
        target = ['cough', 'bark']
        with self.assertRaises(ValueError):
            transform_target = self.multi_transform(target)

    def test_empty_target_no_auto_increment(self):
        """Checks the case when input target is empty with auto_increment=False"""
        target = []
        with self.assertRaises(ValueError):
            transform_target = self.binary_transform_no_auto_increment(target)

    def test_one_intersection_binary_no_auto_increment(self):
        """Checks one intersection with auto_increment=False"""
        target = ['cough']
        transform_target = self.binary_transform_no_auto_increment(target)
        self.assertEqual(transform_target, 0)


if __name__ == "__main__":
    unittest.main()
