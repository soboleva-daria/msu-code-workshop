import numpy as np


class ConvertImage(object):

    def vect_version(self, img, weights):
        return np.sum(img * weights[np.newaxis, np.newaxis, :], axis=2)

    def non_vect_version(self, img, weights):
        height = img.shape[0]
        width = img.shape[1]
        res_img = np.zeros((height, width))
        for ch, w in enumerate(weights):
            for i in range(height):
                for j in range(width):
                    res_img[i, j] += img[i, j, ch] * w
        return res_img

    def medium_version(self, img, weights):
        res_img = np.zeros((img.shape[0], img.shape[1]))
        for ch, w in enumerate(weights):
            res_img += w * img[:, :, ch]
        return res_img
