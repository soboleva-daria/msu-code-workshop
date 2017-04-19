import numpy as np 
import pandas as pd
from cross_validation import mean_squared_error

def compute_pu(Qu, ru, K, lambda_p):
    Au = np.dot(Qu.T, Qu)
    Au.flat[::K + 1] += lambda_p * Qu.shape[0]
    du = np.dot(Qu.T, ru)
    pu = np.linalg.solve(Au, du)
    return pu

def compute_qi(Pi, ri, K, lambda_q):
    Ai = np.dot(Pi.T, Pi)
    Ai.flat[::K + 1] += lambda_q * Pi.shape[0]
    di = np.dot(Pi.T, ri)
    qi = np.linalg.solve(Ai, di)
    return qi

def matrix_factorization(X, movies, users, K=10, N=20, lambda_p=0.2, lambda_q=0.001):
    Q = 0.1 * np.random.random((movies.max() + 1, K))
    P = 0.1 * np.random.random((users.max() + 1, K))
    for n_iter in range(N):
        for uid, idxs in X.groupby('UserID').groups.items():
            P[uid] = compute_pu(Q[X.iloc[idxs].MovieID], X.iloc[idxs].Rating, K, lambda_p)
        for mid, idxs in X.groupby('MovieID').groups.items():
            Q[mid] = compute_qi(P[X.iloc[idxs].UserID], X.iloc[idxs].Rating, K, lambda_q)
    return P, Q

