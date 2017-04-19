from content_based import RidgeRegression
import math
import numpy as np
import pandas as pd

def mean_squared_error(y_true, y_pred, scale=25):
    return np.mean((y_pred - y_true) ** 2) * scale

def train_test_split(ratings, train_frac=0.8):
	train_idxs = []
	test_idxs = []
	for uid, idxs in ratings.sort_values('Timestamp', ascending=False).groupby('UserID').groups.items():
	    train_size = int(math.floor(len(idxs)) * train_frac)
	    train_idxs.extend(idxs[:train_size])
	    test_idxs.extend(idxs[train_size:])

	X_train = ratings[ratings.index.isin(train_idxs)]
	X_test = ratings[ratings.index.isin(test_idxs)]
	return X_train, X_test

def cross_val_score(model, params, X_train, y_train, X_test, y_test):
	errors = []
	for param in params:
		y_pred = model(param).fit(X_train, y_train).predict(X_test)
		errors.append(mean_squared_error(y_pred, y_test))
	return {'best_param': params[np.argmin(errors)], 'error': np.min(errors)}
