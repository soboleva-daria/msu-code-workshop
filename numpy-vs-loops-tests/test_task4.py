import unittest
import numpy as np

from task4 import MaxAfterZero


class TestMaxAfterZero(unittest.TestCase):
    x1 = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
    x2 = np.zeros(10)
    max_after_zero = MaxAfterZero()

    def test_vect_version(self):
        self.assertEqual(
            5,
            self.max_after_zero.vect_version(self.x1)
        )

    def test_non_vect_version(self):
        self.assertEqual(
            5,
            self.max_after_zero.non_vect_version(self.x1)
        )

    def test_smart_version(self):
        self.assertEqual(
            5,
            self.max_after_zero.smart_version(self.x1)
        )

    def test_vect_version_zeros(self):
        self.assertEqual(
            0,
            self.max_after_zero.vect_version(self.x2)
        )

    def test_non_vect_version_zeros(self):
        self.assertEqual(
            0,
            self.max_after_zero.non_vect_version(self.x2)
        )

    def test_smart_version_zeros(self):
        self.assertEqual(
            0,
            self.max_after_zero.smart_version(self.x2)
        )


if __name__ == '__main__':
    unittest.main()
