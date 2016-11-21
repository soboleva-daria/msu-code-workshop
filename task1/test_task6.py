import unittest
import numpy as np
from numpy.testing import assert_equal

from task6 import RunLengthEncode


class TestRunLengthEncode(unittest.TestCase):
    x1 = np.array([2, 2, 2, 3, 3, 3, 5, 2])
    x2 = 6
    run_length_encode = RunLengthEncode()

    def test_vect_version_1darray(self):
        numbers, times_repeat = self.run_length_encode.vect_version(self.x1)
        assert_equal(np.array([2, 3, 5, 2]), numbers)
        assert_equal(np.array([3, 3, 1, 1]), times_repeat)

    def test_smart_version_1darray(self):
        numbers, times_repeat = self.run_length_encode.version_groupby(self.x1)
        assert_equal(np.array([2, 3, 5, 2]), numbers)
        assert_equal(np.array([3, 3, 1, 1]), times_repeat)

    def test_non_vect_version_1darray(self):
        numbers, times_repeat = self.run_length_encode.non_vect_version(
            self.x1)
        assert_equal(np.array([2, 3, 5, 2]), numbers)
        assert_equal(np.array([3, 3, 1, 1]), times_repeat)

    def test_vect_version_0darray(self):
        numbers, times_repeat = self.run_length_encode.vect_version(self.x2)
        assert_equal(np.array([6]), numbers)
        assert_equal(np.array([1]), times_repeat)

    def test_smart_version_0darray(self):
        numbers, times_repeat = self.run_length_encode.version_groupby(self.x2)
        assert_equal(np.array([6]), numbers)
        assert_equal(np.array([1]), times_repeat)

    def test_non_vect_version_0darray(self):
        numbers, times_repeat = self.run_length_encode.non_vect_version(
            self.x2)
        assert_equal(np.array([6]), numbers)
        assert_equal(np.array([1]), times_repeat)


if __name__ == '__main__':
    unittest.main()
