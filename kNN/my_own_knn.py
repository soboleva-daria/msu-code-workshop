import numpy as np
import scipy.sparse as sp


class MyOwnKNN_classifier(object):
    """Classifier implementing the k-nearest neighbors vote.
       My own realization.

    Parameters
    ==========
    k : int, optional (default = 5)
     Number of neighbors to use for queries.

    metric : {'euclidean', 'cosine'} (default = 'euclidean')
        The distance metric to use for the tree.
    """

    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self._fit_method = None

    def fit(self, X, y=None):
        """Fit the model using X as training data.\
        Parameters
        ----------
        X : array-like
            Training data, shape [n_samples, n_features]
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self._fit_method = 'my_own'
        return self

    def kneighbors(self, X, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)

        Returns:
        --------
        dist : numpy array, shape (n_query, k), \
               where k is number of neighbors to get \
               (will use the value passed to the constructor)
           Array representing the lengths to points

        ind : numpy array, shape (n_query, k), \
            Indices of the nearest points in the population matrix.
       """
        if self._fit_method is None:
            raise NotFittedError("Must fit neighbors before querying.")

        X = self._check_array(X, atleast_2d=True)

        train_size = self.X_train.shape[0]
        k = self.k
        if k > train_size:
            raise ValueError(
                "Expected n_neighbors(k) <= n_samples, "
                " but n_samples = {}, n_neighbors = {}".format(train_size,
                                                               n_neighbors)
            )
        n_samples, _ = X.shape
        sample_range = np.arange(n_samples)[:, np.newaxis]

        if self.metric == 'euclidean':
            dist = self.euclidean_distances(X, self.X_train, squared=True)
        else:
            dist = self.cosine_distances(X, self.X_train)

        """Recommended as efficient way to sort a matrix: \
           O(n) time as opposed to full sort that is O(n) * log(n).
           from here::http://stackoverflow.com/questions/10337533/   \
           /a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array
        """

        neigh_ind = np.argpartition(dist, k - 1, axis=1)[:, :k]
        neigh_ind = neigh_ind[sample_range,
                              np.argsort(dist[sample_range, neigh_ind])]
        if return_distance:
            if self.metric == 'euclidean':
                result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
            else:
                if sp.issparse(X):
                    result = dist[
                        sample_range, neigh_ind], self._toarray(neigh_ind)
                else:
                    result = dist[sample_range, neigh_ind], neigh_ind
        else:
            if sp.issparse(X) and self.metric != 'euclidean':
                result = self.toarray(neigh_ind)
            else:
                result = neigh_ind
        return result

    def _toarray(self, matrix):
        tmp_matrix = np.zeros((matrix.shape[0], matrix[0].shape[1]), dtype=int)
        for i in range(matrix.shape[0]):
            tmp_matrix[i] = matrix[i][0]
        return tmp_matrix

    def euclidean_distances(self, X, Y, squared=False):
        """Considering the rows of X as vectors,
           compute the distance matrix between each pair of vectors.
           For efficiency reasons, the euclidean distance between a pair of row
           vector x and y is computed as::
              dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
           source:: sklearn.metrics.pairwise.euclidean_distances

           Parameters
           ----------
           X : array-like, shape (n_samples_1, n_features)
           Y : array-like, shape (n_samples_2, n_features)

           Returns
           -------
           distances : array, shape (n_samples_1, n_samples_2)
        """
        X, Y = self._check_pairwise_arrays(X, Y)

        XX = self.row_norms(X, squared=True)[:, np.newaxis]
        YY = self.row_norms(Y, squared=True)[np.newaxis, :]

        if sp.issparse(X) or sp.issparse(Y):
            dists = (X * Y.T).toarray()
        else:
            dists = np.dot(X, Y.T)
        dists *= -2
        dists += XX
        dists += YY
        dists = np.maximum(dists, 0)
        return dists if squared else np.sqrt(dists)

    def cosine_distances(self, X, Y):
        """Compute cosine distance between samples in X and Y.
           Cosine distance is defined as 1.0 minus the cosine similarity.
           source:: sklearn.metrics.pairwise.cosine_distance

           Parameters
           ----------
           X : array_like, shape (n_samples_1, n_features).
           Y : array_like, shape (n_samples_2, n_features).

           Returns
           -------
           distances : array, shape (n_samples_1, n_samples_2)
        """
        S = self.cosine_similarity(X, Y)
        S *= -1
        S += 1
        return S

    def cosine_similarity(self, X, Y):
        """Compute cosine similarity between samples in X and Y.
           Cosine similarity, or the cosine kernel, computes similarity as the
           normalized dot product of X and Y:
           K(X, Y) = <X, Y> / (||X||*||Y||)
        """
        X, Y = self._check_pairwise_arrays(X, Y)

        X_norms = self.row_norms(X)[:, np.newaxis]
        X_norms[X_norms == 0.0] = 1.0
        X_norm = X / (X_norms)

        Y_norms = self.row_norms(Y)[:, np.newaxis]
        Y_norms[Y_norms == 0.0] = 1.0
        Y_norm = Y / (Y_norms)

        if sp.issparse(X) or sp.issparse(Y):
            return (X_norm * Y_norm.T)
        return np.dot(X_norm, Y_norm.T)

    def row_norms(self, X, squared=False):
        """Row-wise Euclidean norm of X."""
        if sp.issparse(X):
            norm = self.csr_row_norms(X)
            return norm if squared else np.sqrt(norm)

        norm = np.einsum('ij,ij->i', X, X)
        return norm if squared else np.sqrt(norm)

    def csr_row_norms(self, X):
        """Row-wise Euclidean norm of csr matrix X."""
        n_samples = X.shape[0]
        norms = np.zeros(n_samples, dtype=np.float64)
        X_indptr = X.indptr
        X_data = X.data

        for i in range(n_samples):
            sum_ = 0.0
            for j in range(X_indptr[i], X_indptr[i + 1]):
                sum_ += X_data[j] * X_data[j]
                norms[i] = sum_
        return norms

    def _check_pairwise_arrays(self, X, Y):
        """This function first ensures that both X and Y are arrays,\
           and  that the size of the second dimension of the two arrays is equal.
        """
        X = self._check_array(X)
        Y = self._check_array(Y)

        if X.shape[1] != Y.shape[1]:
            raise ValueError("Incompatible dimension for X and Y matrices: "
                             "X.shape[1] == {}\
                         while Y.shape[1] == {}".format(X.shape[1],
                                                        Y.shape[1]
                                                        )
                             )
        return X, Y

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


class NotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting.
       This class inherits from ValueError"""
