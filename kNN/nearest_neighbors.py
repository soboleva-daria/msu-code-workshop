import operator
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sp

from my_own_knn import MyOwnKNN_classifier


class KNN_classifier(object):
    """Classifier implementing the k-nearest neighbors vote.

    Parameters
    ==========
    k : int, optional (default = 5)
     Number of neighbors to use for queries.

    strategy : {'brute', 'ball_tree', 'kd_tree', 'my_own'}, optional,\
               (default = 'brute')
        Strategy used to compute the nearest neighbors:
          - 'ball_tree' will use sklearn.neighbors.NearestNeighbors(algorithm='ball_tree')
          - 'kd_tree' will use sklearn.neighbors.NearestNeighbors(algorithm='kd_tree')
          - 'brute' will use sklearn.neighbors.NearestNeighbors(algorithm='brute')
          - 'my_own' will use MyOwnKnn_Classifier() from my_own_knn

    metric : {'euclidean', 'cosine'} (default = 'euclidean')
        The distance metric to use for the algorithm.

    weights : bool, optional (default = False)
        If provided with True will use a weighted voting.
        A vote per neighbor will be equal to 1/(distance + eps)

    test_block_size : int, optional (default = 1000)
         Size of block to use while searhing for neighbors

    eps : float, optional (default = 1e-5)
        If weights provided with True will be used in the formula for votes
    """

    def __init__(self, k=5, strategy='brute', metric='euclidean',
                 weights=False, test_block_size=1000, eps=1e-5):
        self.k = k
        self.strategy = self._check_strategy(strategy)
        self.metric = self._check_metric(metric)
        self.weights = weights
        self.test_block_size = test_block_size
        self.eps = eps
        if strategy == 'my_own':
            self.alg = MyOwnKNN_classifier(
                k=k,
                metric=metric
            )
        else:
            self.alg = NearestNeighbors(
                n_neighbors=k,
                algorithm=strategy,
                metric=metric,
            )

    def fit(self, X, y=None):
        """Fit the model using X as training data.\
           Will use a strategy passed to the constructor

        Parameters
        ----------
        X : array-like
            Training data, shape [n_samples, n_features]
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        self.alg.fit(self.X_train, self.y_train)
        return self

    def find_kneighbors(self, X, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.
        Will use a strategy passed to the constructor

        Parameters
        ----------
        X : array-like, shape (n_query, n_features).

        return_distance : boolean, optional (default True).
            If False, distances will not be returned

        Returns
        -------
        dist (only present if return_distance=True):
               numpy array, shape (n_query, k), \
               where k is number of neighbors to get \
               (will use the value passed to the constructor)
           Array representing the lengths to points

        ind : numpy array, shape (n_query, k), \
            Indices of the nearest points in the population matrix.
        """
        if self.strategy == 'brute' or self.strategy == 'my_own':
            # 'brute' and 'my_own' strategies store the matrix of pairwise distances
            return self.find_kneighbors_blocks(
                X, return_distance=return_distance)

        return self.alg.kneighbors(X, return_distance=return_distance)

    def find_kneighbors_blocks(self, X, return_distance=True):
        """Please refer to find_kneighbors. The only difference is that\
           find_kneighbors_blocks will find the K-neighbors blocks.
           Will use size of block passed to the constructor"""

        X = self._check_array(X)
        size = self._check_block_size(
            self.test_block_size, X.shape[0])

        k = self.k
        dist = np.empty((0, k))
        ind = np.empty((0, k), dtype=int)
        n_samples = X.shape[0]
        n = int(n_samples / size)

        for idx in range(n):
            neigh_dst, neigh_ind = self.alg.kneighbors(
                X[idx * size:(idx + 1) * size])
            dist = np.vstack((dist, neigh_dst))
            ind = np.vstack((ind, neigh_ind))

        if n_samples % size != 0:
            idx = n - 1
            neigh_dst, neigh_ind = self.alg.kneighbors(X[(idx + 1) * size:])
            dist = np.vstack((dist, neigh_dst))
            ind = np.vstack((ind, neigh_ind))

        """for block in np.array_split(X, int(X.shape[0]/self.test_block_size)):
            neigh_dst, neigh_ind = self.alg.kneighbors(block)
            dist = np.vstack((dist, neigh_dst))
            ind = np.vstack((ind,  neigh_ind))"""

        if return_distance:
            return dist, ind
        return ind

    def predict_klist(self, X_test, neigh_dist, neigh_ind, k_list):
        res_k = []
        for k in k_list:
            res_k.append(
                self.predict(
                    X_test, neigh_dist[
                        :, :k], neigh_ind[
                        :, :k]))
        return res_k

    def predict(self, X, neigh_dist=None, neigh_ind=None):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features).
            Test samples.

        dist : numpy array, optional (default=None)
               array representing the lengths to points.

        ind : numpy array, optional (default=None)
              indices of the nearest points in the population matrix.

        Returns
        -------
        y : numpy array of shape [n_query]
            Class labels for each data sample.
        """
        if self.weights:
            return self.weighted_predict(X, neigh_dist, neigh_ind)

        X = self._check_array(X, atleast_2d=True)

        if neigh_dist is None:
            neigh_dist, neigh_ind = self.find_kneighbors(X)
        n_query = X.shape[0]
        y_pred = []
        y_train = self.y_train

        for i in range(n_query):
            ind = neigh_ind[i]
            classes_ = y_train[ind]
            y_pred.append(np.bincount(classes_).argmax())

        return np.array(y_pred, dtype=self.y_train[0].dtype)

    def weighted_predict(self, X, neigh_dist=None, neigh_ind=None):
        """Please refer to predict. The difference is that\
           weights will be used in prediction.
           A vote per neighbor will be equal to 1/(distance + eps)
        """

        X = self._check_array(X, atleast_2d=True)
        
        if neigh_dist is None:
            neigh_dist, neigh_ind = self.find_kneighbors(X)

        n_samples = X.shape[0]
        y_train = self.y_train
        y_pred = []
        for i in range(n_samples):
            ind = neigh_ind[i]
            dst = neigh_dist[i]

            classes_ = y_train[ind]
            cls_votes = {}
            eps = self.eps
            for d, cls in zip(dst, classes_):
                if cls not in cls_votes:
                    cls_votes[cls] = 1 / (d + eps)
                else:
                    cls_votes[cls] += 1 / (d + eps)
            y_pred.append(
                max(cls_votes.items(), key=operator.itemgetter(1))[0])

        return np.array(y_pred, dtype=self.y_train[0].dtype)

    def _check_strategy(self, strategy):
        """Check to make sure strategy is valid"""
        if strategy in ('brute', 'ball_tree', 'kd_tree', 'my_own'):
            return strategy
        else:
            raise ValueError(
                "strategy not recognized: should be 'brute','ball_tree', or 'kd_tree'")

    def _check_metric(self, metric):
        """Check to make sure metric is valid"""
        if metric in ('euclidean', 'cosine'):
            return metric
        else:
            raise ValueError(
                "metric not recognized: should be 'euclidean' or 'cosine'")

    def _check_block_size(self, test_block_size, N):
        """Check to make sure test_block_size is greater than zero"""
        if test_block_size > 0 and test_block_size <= N:
            return test_block_size
        else:
            raise ValueError("test_block_size should be in range (0, {}]".format(N))

    def _check_array(self, array, atleast_2d=False):
        """Input validation on an array, list, sparse matrix
        Parameters
        ----------
        array : array-like
           Input object to check / convert.
        ensure_2d : boolean, optional (default=False)
           Whether to make X at least 2d."""

        if sp.issparse(array):
            if atleast_2d and array.ndim < 2:
                return np.atleast_2d(array)
            return array

        if not isinstance(array, np.ndarray):
            array = np.array(array)

        if atleast_2d and array.ndim < 2:
            array = np.atleast_2d(array)
        return array
