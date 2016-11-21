import numpy as np


class MaxAfterZero(object):

    def vect_version(self, x):
        return np.max(x[np.roll(x == 0, 1)][1:])

    def non_vect_version(self, x):
        return max(
            [x[i + 1] for i, elem in enumerate(x[:-1])
             if elem == 0
             ])
`
    def smart_version(self, x):
        return max(map(lambda p: p[1], filter(
            lambda p: p[0] == 0, zip(x, x[1:]))))
