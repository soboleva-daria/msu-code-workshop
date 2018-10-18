import numpy as np
from collections import Counter


class MultiSet(object):

    def vect_version(self, x, y):
        if np.ndim(x) == 0 or np.ndim(y) == 0:
            return x == y
        return np.all(np.sort(x) == np.sort(y))

    def non_vect_version(self, x, y):
        if np.ndim(x) == 0 or np.ndim(y) == 0:
            return x == y
        return Counter(x) == Counter(y)

    def another_vect_version(self, x, y):
        if np.ndim(x) == 0 or np.ndim(y) == 0:
            return x == y

        vals1, counts1 = np.unique(x, return_counts=True)
        vals2, counts2 = np.unique(y, return_counts=True)
        return np.all(vals1 == vals2) and np.all(counts1 == counts2)
