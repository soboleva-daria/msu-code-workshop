import unittest
import numpy as np
from numpy.testing import assert_equal

from task2 import MatrixFromVectors


class TestMatrixFromVectors(unittest.TestCase):
    i1 = np.arange(2)
    j1 = np.arange(2) * 2
    X1 = np.random.rand(10, 5)
    i2 = 1
    j2 = 4
    X2 = np.eye(5)
    matrix_from_vect = MatrixFromVectors()

    def test_vect_version_hstack_1darray(self):
        assert_equal(
            self.X1[self.i1, self.j1],
            self.matrix_from_vect.vect_version_hstack(
                self.X1,
                self.i1,
                self.j1
            )
        )

    def test_vect_version_transpose_1darray(self):
        assert_equal(
            self.X1[self.i1, self.j1],
            self.matrix_from_vect.vect_version_transpose(
                self.X1,
                self.i1,
                self.j1
            )
        )

    def test_vect_version_concat_1darray(self):
        assert_equal(
            self.X1[self.i1, self.j1],
            self.matrix_from_vect.vect_version_concat(
                self.X1,
                self.i1,
                self.j1
            )
        )

    def test_non_vect_version_1darray(self):
        assert_equal(
            self.X1[self.i1, self.j1],
            self.matrix_from_vect.non_vect_version(
                self.X1,
                self.i1,
                self.j1
            )
        )

    def test_fast_version_1darray(self):
        assert_equal(
            self.X1[self.i1, self.j1],
            self.matrix_from_vect.fast_version(
                self.X1,
                self.i1,
                self.j1
            )
        )

    def test_vect_version_hstack_0darray(self):
        self.assertEqual(
            self.X2[self.i2, self.j2],
            self.matrix_from_vect.vect_version_hstack(
                self.X2,
                self.i2,
                self.j2
            )
        )

    def test_vect_version_transpose_0darray(self):
        self.assertEqual(
            self.X2[self.i2, self.j2],
            self.matrix_from_vect.vect_version_transpose(
                self.X2,
                self.i2,
                self.j2
            )
        )

    def test_vect_version_concat_0darray(self):
        self.assertEqual(
            self.X2[self.i2, self.j2],
            self.matrix_from_vect.vect_version_concat(
                self.X2,
                self.i2,
                self.j2
            )
        )

    def test_non_vect_version_0darray(self):
        self.assertEqual(
            self.X2[self.i2, self.j2],
            self.matrix_from_vect.non_vect_version(
                self.X2,
                self.i2,
                self.j2
            )
        )

    def test_fast_version_0darray(self):
        self.assertEqual(
            self.X2[self.i2, self.j2],
            self.matrix_from_vect.fast_version(
                self.X2,
                self.i2,
                self.j2
            )
        )

if __name__ == '__main__':
    unittest.main()
