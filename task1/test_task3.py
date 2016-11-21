import unittest
import numpy as np

from task3 import MultiSet


class TestMultiSet(unittest.TestCase):

    x1 = np.array([1, 2, 2, 4])
    y1 = np.array([1, 2, 4])

    x2 = 1
    y2 = 1
    multiset = MultiSet()

    def test_vect_version_1darray(self):
        self.assertEqual(
            False,
            self.multiset.vect_version(
                self.x1,
                self.y1
            )
        )

    def test_non_vect_version_1darray(self):
        self.assertEqual(
            False,
            self.multiset.non_vect_version(
                self.x1,
                self.y1
            )
        )

    def test_another_vect_version_1darray(self):
        self.assertEqual(
            False,
            self.multiset.another_vect_version(
                self.x1,
                self.y1
            )
        )

    def test_vect_version_0darray(self):
        self.assertEqual(
            True,
            self.multiset.vect_version(
                self.x2,
                self.y2
            )
        )

    def test_non_vect_version_0darray(self):
        self.assertEqual(
            True,
            self.multiset.non_vect_version(
                self.x2,
                self.y2
            )
        )

    def test_another_vect_version_0darray(self):
        self.assertEqual(
            True,
            self.multiset.another_vect_version(
                self.x2,
                self.y2
            )
        )

if __name__ == '__main__':
    unittest.main()
