import numpy as np
from functools import reduce
import operator


class ProdNonZero(object):

    def vect_version(self, X):
        if np.ndim(X) == 0:
            return X
        if np.ndim(X) == 1:
            return np.prod(X[np.nonzero(X)])
        X_diag = np.diag(X)
        return np.prod(X_diag[np.nonzero(X_diag)])

    def non_vect_version(self, X):
        if np.ndim(X) == 0:
            return X
        if np.ndim(X) == 1:
            return reduce(operator.mul, [x for x in X if x != 0])
        return reduce(operator.mul,
                      [X[i, i] for i in range(min(X.shape[0], X.shape[1]))
                       if X[i, i] != 0])

    def medium_version(self, X):
        if np.ndim(X) == 0:
            return X
        if np.ndim(X) == 1:
            return reduce(operator.mul, [x for x in X if x != 0])
        X_diag = np.diag(X)
        return reduce(operator.mul, [x for x in X_diag if x != 0])
