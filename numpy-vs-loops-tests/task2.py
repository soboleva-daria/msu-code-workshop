import numpy as np


class MatrixFromVectors(object):

    def vect_version_hstack(self, X, i, j):
        if np.ndim(i) == 0:
            return X[i, j]
        idx = np.hstack((i[:, np.newaxis], j[:, np.newaxis]))
        return X[idx[:, 0], idx[:, 1]]

    def vect_version_transpose(self, X, i, j):
        if np.ndim(i) == 0:
            return X[i, j]
        idx = np.array([i, j]).T
        return X[idx[:, 0], idx[:, 1]]

    def vect_version_concat(self, X, i, j):
        if np.ndim(i) == 0:
            return X[i, j]
        idx = np.concatenate((i[:, np.newaxis], j[:, np.newaxis]), axis=1)
        return X[idx[:, 0], idx[:, 1]]

    def non_vect_version(self, X, i, j):
        if np.ndim(i) == 0:
            return X[i, j]

        return np.array([X[ind1, ind2] for ind1, ind2 in zip(i, j)])

    def fast_version(self, X, i, j):
        return X[i, j]
