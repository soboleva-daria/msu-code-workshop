import numpy as np
from scipy.spatial import distance
from numpy import sqrt, sum


class EuclideanDist(object):

    def vect_version(self, X, Y):
        X_dim = np.ndim(X)
        Y_dim = np.ndim(Y)
        if X_dim == 0 and Y_dim == 0:
            return sqrt((X - Y)**2)
        if X_dim == 1 and Y_dim == 1:
            return sqrt(sum((X - Y)**2))
        return sqrt(
            np.sum((X[:, np.newaxis, np.newaxis] - Y[:, np.newaxis])**2, axis=3))[:, :, 0]

    def non_vect_version(self, X, Y):
        X_dim = np.ndim(X)
        Y_dim = np.ndim(Y)
        if X_dim == 0 and Y_dim == 0:
            return sqrt((X - Y)**2)
        if X_dim == 1 and Y_dim == 1:
            return sqrt(sum((X - Y)**2))

        X_res = np.zeros((X.shape[0], Y.shape[0]))
        for i, object_X in enumerate(X):
            for j, object_Y in enumerate(Y):
                X_res[i, j] = sqrt(sum([(x1 - x2)**2
                                        for x1, x2 in zip(object_X, object_Y)]))
        return X_res

    def version_scipy(self, X, Y):
        X_dim = np.ndim(X)
        Y_dim = np.ndim(Y)
        if X_dim == 0 and Y_dim == 0:
            return sqrt((X - Y)**2)
        if X_dim == 1 and Y_dim == 1:
            return sqrt(sum((X - Y)**2))

        X_res = np.zeros((X.shape[0], Y.shape[0]))
        for i, object_X in enumerate(X):
            for j, object_Y in enumerate(Y):
                X_res[i, j] = distance.euclidean(object_X, object_Y)
        return X_res
