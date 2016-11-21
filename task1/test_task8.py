import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import multivariate_normal

from task8 import MultiLogNormDensity


class TestMultiLogNormDensity(unittest.TestCase):
    X = np.random.rand(200, 10)
    mu = np.mean(X, axis=0)
    sigma = np.eye(10)
    logpdf = multivariate_normal(mu, sigma).logpdf(X)
    ml_logpdf = MultiLogNormDensity()

    def test_vect_version(self):
        assert_almost_equal(
            self.logpdf,
            self.ml_logpdf.vect_version(self.X, self.mu, self.sigma)

        )

    def test_version_non_vect_version(self):
        assert_almost_equal(
            self.logpdf,
            self.ml_logpdf.non_vect_version(self.X, self.mu, self.sigma)
        )

    def test_another_vect_version(self):
        assert_almost_equal(
            self.logpdf,
            self.ml_logpdf.another_vect_version(
                self.X,
                self.mu,
                self.sigma))


if __name__ == '__main__':
    unittest.main()
