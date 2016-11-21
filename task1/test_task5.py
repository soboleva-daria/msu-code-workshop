import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.misc import imread

from task5 import ConvertImage


class TestConvertImage(unittest.TestCase):
    src_img = np.load('src_img.npy')
    res_img = np.load('res_img.npy')
    weights = np.array([0.299, 0.587, 0.114])
    convert_image = ConvertImage()

    def test_vect_version(self):
        assert_almost_equal(
            self.res_img,
            self.convert_image.vect_version(self.src_img, self.weights)
        )

    def test_non_vect_version(self):
        assert_almost_equal(
            self.res_img,
            self.convert_image.non_vect_version(self.src_img, self.weights)
        )

    def test_medium_version(self):
        assert_almost_equal(
            self.res_img,
            self.convert_image.medium_version(self.src_img, self.weights)
        )


if __name__ == '__main__':
    unittest.main()
