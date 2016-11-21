import numpy as np
from numpy import linalg, log
from math import pi


class MultiLogNormDensity(object):

    def matrix_vect_multiply(self, X, vect):
        if np.ndim(X) == 1:
            return np.array(
                ([sum(x * y for x, y in zip(X, vect))])
            )
        return np.array(
            [sum(x * y for x, y in zip(X_row, vect)) for X_row in X]
        )

    def matrix_vect_subtract(self, X, vect):
        return np.array(
            [[X[obj_j, i] - vect_i
              for i, vect_i in enumerate(vect)]
             for obj_j in range(X.shape[0])]
        )

    def vect_version(self, X, mu, sigma):
        N = X.shape[0]
        D = X.shape[1]

        inv = linalg.inv(sigma)
        _, logdet = linalg.slogdet(sigma)
        const = D * log(2 * pi) + logdet
        X_mu = X - mu
        right_part = np.dot(X_mu, inv.T)
        return -0.5 * (const + (X_mu * right_part).sum(axis=1))

    def non_vect_version(self, X, mu, sigma):
        N = X.shape[0]
        D = X.shape[1]

        inv = linalg.inv(sigma)
        _, logdet = linalg.slogdet(sigma)
        const = -0.5 * (D * log(2 * pi) + logdet)

        exp_obj = []
        X_mu = self.matrix_vect_subtract(X, mu)
        for obj in range(N):
            right_part = self.matrix_vect_multiply(
                inv,
                X_mu[obj]
            )
            prod = self.matrix_vect_multiply(
                X_mu[obj],
                right_part
            )
            exp_obj.append(prod)
        return const + np.array([-0.5 * i for i in exp_obj])[:, 0]

    def another_vect_version(self, X, mu, sigma):
        N = X.shape[0]
        D = X.shape[1]

        inv = linalg.inv(sigma)
        _, logdet = linalg.slogdet(sigma)
        const = D * log(2 * pi) + logdet
        X_mu = X - mu
        return -0.5 * (const + (np.diag(np.dot(X_mu, (np.dot(X_mu, inv)).T))))
