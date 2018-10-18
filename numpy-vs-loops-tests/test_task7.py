import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.spatial import distance

from task7 import EuclideanDist


class TestEuclideanDist(unittest.TestCase):
    X0 = 1
    Y0 = 2
    X1 = np.random.rand(100)
    Y1 = np.random.rand(100)
    X2 = np.random.rand(200, 5)
    Y2 = np.random.rand(100, 5)
    dist0 = 1.0
    dist1 = np.sqrt(sum((X1 - Y1)**2))
    distance = distance.cdist(X2, Y2, 'euclidean')
    euclidean_dist = EuclideanDist()

    def test_vect_version_0darray(self):
        self.assert_almost_equal(
            self.dist0,
            self.euclidean_dist.vect_version(self.X0, self.Y0)
        )

    def test_version_scipy_0darray(self):
        self.assert_almost_equal(
            self.dist0.
            self.euclidean_dist.version_scipy(self.X0, self.Y0)
        )

    def test_non_vect_version_0darray(self):
        self.assert_almost_equal(
            self.dist0,
            self.euclidean_dist.non_vect_version(self.X0, self.Y0)
        )

    def test_vect_version_1darray(self):
        self.assert_almost_equal(
            self.dist1,
            self.euclidean_dist.vect_version(self.X1, self.Y1)
        )

    def test_version_scipy_1darray(self):
        self.assert_almost_equal(
            self.dist1,
            self.euclidean_dist.version_scipy(self.X1, self.Y1)
        )

    def test_non_vect_version_1darray(self):
        self.assert_almost_equal(
            self.dist1,
            self.euclidean_dist.non_vect_version(self.X1, self.Y1)
        )

    def test_vect_version_2darray(self):
        assert_almost_equal(
            self.distance,
            self.euclidean_dist.vect_version(self.X2, self.Y2)
        )

    def test_version_scipy_2darray(self):
        assert_almost_equal(
            self.distance,
            self.euclidean_dist.version_scipy(self.X2, self.Y2)
        )

    def test_non_vect_version_2darray(self):
        assert_almost_equal(
            self.distance,
            self.euclidean_dist.non_vect_version(self.X2, self.Y2)
        )


if __name__ == '__main__':
    unittest.main()
