import numpy as np
import scipy.sparse as sp
from nearest_neighbors import KNN_classifier


class CrossValidation(object):
    """Class to implement cross_validation.

       Parameters
       ==========
       shuffle : boolean, optional (default=False)
            Whether to shuffle the data before splitting into batches.
    """

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def kfold(self, n, n_folds=3):
        """Provides train/test indices to split data in train test sets. Split
           dataset into k consecutive folds (without shuffling by default).
           The first n % n_folds folds have size n // n_folds + 1, other folds have
           size n // n_folds.
        """
        if n_folds <= 1:
            raise ValueError(
                "k-fold cross validation requires at least one"
                " train / validation split by setting n_folds=2 or more,"
                " got n_folds={}.".format(n_folds))

        if n_folds > n:
            raise ValueError(
                ("Cannot have number of folds n_folds={} greater"
                 " than the number of samples: {}.").format(n_folds, n))

        idxs = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idxs)

        fold_sizes = (n // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n % n_folds] += 1
        curr = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = curr, curr + fold_size
            valid = idxs[start:stop]
            train = idxs[~np.in1d(idxs, valid)]
            folds.append((np.array(train), np.array(valid)))
            curr = stop

        return folds

    def knn_cross_val_score(self, X, y, k_list=[3], score='accuracy', cv=None,
                            weights=False, metric='euclidean',
                            strategy='brute', test_block_size=1000):
        """Evaluate a score by cross-validation
        Parameters
        ----------
        X : array-like
            Training data.

        y : array-like
            The target variable to try to predict.

        k_list : list, optional (default=[3])
            The list of check values for the number of nearest neighbors.
            (Must be sorted in ascending order)

        score : {'accuracy'}
           Score method, providing an evaluation criterion of different models.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.

        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - An iterable, the output of kfold function.

        metric : {'euclidean', 'cosine'} (default = 'euclidean')
            The distance metric to use for the algorithm.

        weights : bool, optional (default = False)
            If provided with True will use a weighted voting.
            A vote per neighbor will be equal to 1/(distance + eps)

        test_block_size : int, optional (default = 1000)
            Size of block to use while searhing for neighbors

        Returns:
        -------
        A dictionary where keys are values from k_list and elements\
        are numpy arrays with the quality at every fold.
        """

        cv = self._check_cv(cv, X)
        klist = self._check_klist(k_list)
        score = self._check_scoring(score)
        scores = {}

        k_max = max(k_list)
        estimator = KNN_classifier(
            k_max,
            strategy,
            metric,
            weights,
            test_block_size
        )

        for train, test in cv:
            score_k = \
                self._fit_and_score(estimator, X, y,
                                    score, train, test,
                                    k_list
                                    )
            for k, score in zip(k_list, score_k):
                if k not in scores:
                    scores[k] = []
                scores[k].append(score)

        for k in k_list:
            scores[k] = np.array(scores[k])
        return scores

    def _fit_and_score(self, estimator, X, y, score,
                       train_idx, test_idx, k_list):
        """Fit estimator and compute scores for a given dataset split.

           Parameters
           ----------
           estimator : estimator object implementing 'fit'
                     The object to use to fit the data.
           X : array-like
               The data to fit.
           y : array-like
               The target variable to try to predict.
           score : {'accuracy'}
               Score method, providing an evaluation criterion of different models.
           train : array-like, shape (n_train_samples)
                Indices of training samples.
           test  : array-like, shape (n_test_samples,)
                Indices of test samples.

        Returns
        -------
        scores : list of scores for each k
        """

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        estimator.fit(X_train, y_train)

        neigh_dist, neigh_ind = estimator.find_kneighbors(X_test)

        res_k = estimator.predict_klist(X_test, neigh_dist, neigh_ind, k_list)

        scores = []
        for res in res_k:
            scores.append(accuracy_score(y_test, res))
        return scores

    def _check_cv(self, cv, X):
        """Input checker utility for building a CV.
           If cv is None, use the default 3-fold cross-validation
        """
        if cv is None:
            cv = self.kfold(X.shape[0], 3)
        return cv

    def _check_klist(self, k_list):
        """Checks whether k_list is a list, also
           checks whether k_list is not empty,
           otherwise pass klist=[3].

           Return sorted k_list.
        """
        if not isinstance(k_list, list):
            k_list = list(k_list)
        if len(k_list) < 1:
            k_list = [3]
        return sorted(k_list)

    def _check_scoring(self, score):
        """Check to make sure score is valid"""
        if score == 'accuracy':
            return score
        raise ValueError("score not recognized: should be 'accuracy'")


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
       (1/n_samples*sum(y_true==y_pred))

       Paramters:
       ---------
       y_true : 1d array-like
            Correct labels.
       y_pred : 1d array-like
            Predicted labels, as returned by a classifier.

       Returns:
       -------
       score : float
       """
    if y_true.size != y_pred.size:
        raise ValueError("Found arrays with inconsistent numbers of samples:\
                             size of y_true:{}, size of y_pred:{}")
    score = y_true == y_pred
    return np.average(score)
