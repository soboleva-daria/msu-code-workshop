import unittest
import numpy as np

from task1 import ProdNonZero


class TestProdNonZero(unittest.TestCase):
    X1 = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]])
    X2 = np.arange(5)
    X3 = 1
    prod_non_zero = ProdNonZero()

    def test_medium_version_X_2darray(self):
        self.assertEqual(3, self.prod_non_zero.medium_version(self.X1))

    def test_non_vect_version_X_2darray(self):
        self.assertEqual(3, self.prod_non_zero.non_vect_version(self.X1))

    def test_vect_version_X_2darray(self):
        self.assertEqual(3, self.prod_non_zero.vect_version(self.X1))

    def test_vect_version_X_1darray(self):
        self.assertEqual(24, self.prod_non_zero.vect_version(self.X2))

    def test_non_vect_version_X_1darray(self):
        self.assertEqual(24, self.prod_non_zero.non_vect_version(self.X2))

    def test_medium_version_X_1darray(self):
        self.assertEqual(24, self.prod_non_zero.medium_version(self.X2))

    def test_vect_version_X_0darray(self):
        self.assertEqual(1, self.prod_non_zero.vect_version(self.X3))

    def test_non_vect_version_X_0darray(self):
        self.assertEqual(1, self.prod_non_zero.non_vect_version(self.X3))

    def test_medium_version_X_0darray(self):
        self.assertEqual(1, self.prod_non_zero.medium_version(self.X3))


if __name__ == '__main__':
    unittest.main()
