import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
from cvxopt import matrix
from cvxopt.solvers import qp, options
import time
import matplotlib.pyplot as plt


def visualize(X, y, alg_svm, show_vectors=False):
    """This method is provided in order to visualize
    objects from X matrix, support vectors from alg_svm class and
    separating surface (only for dual alg_svm).
    Available only in 2-dim space.

    Parameters:
    ----------
    X : numpy.array, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and

    y : numpy.array, shape = [n_samples, 1]
        Target vector relative to X

    alg_svm : object of class SVM. (Must be fitted)

    show_vectors : optional (default=False)
       if provided with True, plot support vectors also. (Only in dual task)
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Incompatible type of vector. Must be numpy.array")

    if not isinstance(y, np.ndarray):
        raise ValueError("Incompatible type of vector. Must be numpy.array")

    y = y.reshape(y.shape[0], )

    if X.ndim != 2:
        raise ValueError('visualization is only available in 2-dim space')

    if alg_svm.fit_method is None:
        raise NotFittedError("Must fit SVM before querying.")

    plt.figure(figsize=(9, 6))
    plt.xlabel('x', fontsize = 18)
    plt.ylabel('y', fontsize = 18)
    plt.title('Model data', fontsize = 18)

    visualize_sep_hyperline(alg_svm, X)
    
    if alg_svm.method in ('dual', 'libsvm'):
        if show_vectors:
            plt.scatter(alg_svm.sv[:, 0], alg_svm.sv[:, 1], s=80,
                        facecolors='none', zorder=10)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.show()


def visualize_sep_hyperline(alg_svm, X):
    """This method is provided in order to visualize separating surface.

    Parameters:
    ----------
    alg_svm : object of class SVM. (Must be fitted)

    X : numpy.array, shape = [n_samples, n_features]
      Training vector, where n_samples in the number of samples and
      n_features is the number of features.
    """
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = alg_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

EPS = 1e-10
class SVM(object):
    """Class implementing Support Vector Classification.

     Parameters
     ----------
         C : float, optional (default=1.0)
           Penalty parameter C of the error term.

         method : string, optional (default='dual')
           Specifies the method to be used in the algorithm.
           It must be one of 'primal', 'dual', 'subgradient', 'stoch_subgradient', 'liblinear' or
           'libsvm'.

        kernel : string, optional (default=None)
          Specifies the kernel type to be used in the algorithm.
          It must be linear or rbf.

        gamma : float, optional (default=None)
          Kernel coefficient for rbf.
          If None, defaults to 1.0 / n_samples.
     """

    def __init__(
            self,
            C=1.0,
            method='dual',
            kernel=None,
            gamma=None):
        self.C = C
        self.method = method
        self.fit_method = None
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y,
            tol=1e-4, max_iter=1000, verbose=False,
            stop_criterion='objective', batch_size=1,
            lamb=0.5, alpha=0.5, beta=0.5, min_iter=10):
        """Fit the model according to the given training data.

        Parameters:
        ----------
        X : numpy.array, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : numpy.array, shape = [n_samples, 1]
            Target vector relative to X

        tol : float, optional (default=1e-4)
              Tolerance for stopping criterion.

        max_iter : int, (default=1000)
               The maximum number of iterations to be run.

        verbose : bool, optional (default=False)
               Enable verbose output.

        stop_criterion : string, optional (default='objective')
              The stop criteria. Must be 'objective' or 'argument'.
                -- 'objective' : the value of the objective function.
                -- 'argument'  : the value of the weights norm.

        batch_size : int, optional (default=1)
              Size of the batch for training in SGD.

        alpha : optional, float (default=0.5)
               Parameter for step in subgradient methods.

        beta : optional, float (default=0.5)
               Parameter for step in subradient methods.
               
        min_iter : int, (default=10)
               The minimum number of iterations to be run in SGD.

        Returns:
        -------
        The dictionary with fields:
            -- status : the reason for stopping the algorithm.
                 0  - the stop criteria,
                 1  - the achievement of max_iter.
            -- objective_curve : list of values of the objective function.
                   Only for 'subgradient' and 'stoch_subgradient' methods.
            -- time : training time.
        """ 
        method = self._check_method(self.method)
        
        if method == 'primal':
            return self._fit_primal(X, y, tol, max_iter, verbose)

        if method == 'dual':
            return self._fit_dual(X, y, tol, max_iter, verbose)
        
        if method == 'subgradient':
            return self._fit_subgradient(X,
                                         y,
                                         tol,
                                         max_iter,
                                         verbose,
                                         stop_criterion,
                                         alpha, 
                                         beta)
            
        if method == 'stoch_subgradient':
            return self._fit_stoch_subgradient(
                                           X,
                                           y, 
                                           tol, 
                                           max_iter, 
                                           verbose, 
                                           stop_criterion, 
                                           batch_size, 
                                           lamb, 
                                           alpha, 
                                           beta, 
                                           min_iter)

        if method == 'liblinear':
            return self._fit_liblinear(X, y, tol, max_iter, verbose)

        if method == 'libsvm':
            return self._fit_libsvm(X, y, tol, max_iter, verbose)

    def _fit_primal(self, X, y, tol, max_iter, verbose):
        """Primal SVM.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
            y : numpy.array, shape = [n_samples]
                Target vector relative to X

           tol : float
              Tolerance for stopping criterion.

           max_iter : int
               The maximum number of iterations to be run.

           verbose : bool 
               Enable verbose output

        Finds:
        -----
            w (weights): numpy.array, shape = [D]
                Solution of the primal SVM.

        Returns:
        -------
            The dictionary with fields:
            -- status : the reason for stopping the algorithm
                 0  - the stop criteria,
                 1  - the achievement of max_iter.
            -- time : training time.

        """
        if self.method != 'primal':
            raise ValueError(
                  '_fit_primal method is not available. Please, check your method parameter')
            
        self.C = self._check_C(self.C)
        self.max_iter = self._check_max_iter(max_iter)
        X, y = self._check_X_y(X, y) 
        
        self.fit_method = self.method 
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        
        N, D = X.shape
        P = np.zeros((D + N + 1, D + N + 1))
        for i in range(D):
            P[i, i] = 1
        q = np.vstack([np.zeros((D + 1, 1)), self.C * np.ones((N, 1))])

        G = np.zeros((2 * N, N + D + 1))
        G[:N, :D] = X * y[:, np.newaxis]
        G[:N, D] = y.T
        G[:N, D + 1:] = np.eye(N)
        G[N:, D + 1:] = np.eye(N)
        G = -G

        h = np.zeros((2 * N, 1))
        h[:N] = -1

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)

        options['maxiters'] = self.max_iter
        options['show_progress'] = self.verbose
        options['abstol'] = self.tol
        
        start = time.clock()
        solution = qp(P, q, G, h)

        self.w = np.ravel(solution['x'])[:D + 1]
        self.w0 = self.w[0]
        self.w = self.w[1:]
        status = int(solution['iterations'] >= options['maxiters'])
        
        return {'status': status,
                'time': time.clock() - start}

    def _fit_dual(self, X, y, tol, max_iter, verbose):
        """Dual SVM.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
            y : numpy.array, shape = [n_samples]
                Target vector relative to X

           tol : float
              Tolerance for stopping criterion.

           max_iter : int
               The maximum number of iterations to be run.

           verbose : bool 
               Enable verbose output
        Finds:
        ------
            A (values of the dual variables): numpy.array, shape = [N]
                Solution of the dual SVM.

        Returns:
        ------- 
            The dictionary with fields:
            -- status : the reason for stopping the algorithm
                 0  - the stop criteria,
                 1  - the achievement of max_iter.
            -- time : training time.
        """
        
        if self.method != 'dual':
            raise ValueError(
                  '_fit_dual method is not available. Please, check your method parameter') 
            
        self.C = self._check_C(self.C)
        self.max_iter = self._check_max_iter(max_iter)
        self.kernel = self._check_kernel(self.kernel)
        X, y = self._check_X_y(X, y)
        self.kernel = self._check_kernel(self.kernel)
        self.gamma = self._check_gamma(self.gamma, X.shape[1])
        
        self.fit_method = self.method
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        
        if self.kernel == 'linear':
            K = self.linear_kernel(X, X)
        else:
            K = self.rbf_kernel(X, X, self.gamma) 
        N, D = X.shape

        P = matrix(np.outer(y,y) * K)
        q = matrix(np.ones(N) * -1)
        A = matrix(y, (1, N), tc='d')
        b = matrix(0.0)

        tmp1 = np.diag(np.ones(N) * -1)
        tmp2 = np.identity(N)
        G = matrix(np.vstack((tmp1, tmp2)))

        tmp1 = np.zeros(N)
        tmp2 = np.ones(N) * self.C
        h = matrix(np.hstack((tmp1, tmp2)))
        
        options['maxiters'] = self.max_iter
        options['show_progress'] = self.verbose
        options['abstol'] = self.tol
        
        start = time.clock()
        solution = qp(P, q, G, h, A, b)

        self.A = np.ravel(solution['x'])

        self.ind_sv = self.A > EPS
        self.sv = X[self.ind_sv]
        self.sv_y = y[self.ind_sv]

        status = int(solution['iterations'] >= options['maxiters'])

        return {'status' : status, 
                'time' : time.clock() - start}

    def compute_primal_objective(self, X, y):
        """Finds the objective function in primal SVM.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
            y : numpy.array, shape = [n_samples, 1]
                Target vector relative to X

        Returns:
        ------- 
            Value of the objective function in primal SVM.
        """
        if self.fit_method is None:
            raise NotFittedError("Must fit svm before querying.")
    
        if self.method not in ('primal', 'subgradient', 'stoch_subgradient', 
                                'liblinear'):
            raise ValueError ("You can't call primal methods for dual task")
            
        X, y = self._check_X_y(X, y)

        return 0.5 * np.linalg.norm(self.w) + self.C * np.sum(
                (1 - ((X * self.w).sum(axis=1) + self.w0) * y).clip(min=0))

    def compute_dual_objective(self, X, y):
        """Finds the objective function in dual SVM.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
            y : numpy.array, shape = [n_samples, 1]
                Target vector relative to X

        Returns:
        ------- 
            Value of the objective function in dual SVM.
            """
        if self.fit_method is None:
            raise NotFittedError("Must fit svm before querying.")

        if self.method not in ('dual', 'libsvm'):
                raise ValueError ("You can't call dual methods for primal task")

        X, y = self._check_X_y(X, y)
        
        if self.kernel == 'linear':
                K = self.linear_kernel(X, X)
        else:
                K = self.rbf_kernel(X, X, self.gamma)  
        return -np.sum(self.A) + 0.5 * \
             np.sum(np.outer(self.A, self.A) * np.outer(y, y) * K)

    def compute_primal_objective_ind(self, X, y, ind):
        """The same as compute_primal_objective method.
        The only difference is that in this function we also return indices 
        of elements in X with nonzero indicator in subgradient of loss function.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
            y : numpy.array, shape = [n_samples]
                Target vector relative to X
                
            ind : numpy.array, shape = [n_samples_chosen, n_features]
                Indices of random chosen samples.

        Returns:
        ------- 
            Value of the objective function in primal SVM and 
            indices of elements in X with nonzero indicator in subgradient of loss function
        """
        
        loss = (1 - ((X[ind] * self.w).sum(axis=1) + self.w0) * y[ind]).clip(min=0)

        return 0.5 * np.linalg.norm(self.w) + self.C * \
            np.sum(loss), ind[np.nonzero(loss)]

    def compute_subgradient(self, X, y):
        """Compute subgradient of loss function.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
            y : numpy.array, shape = [n_samples]
                Target vector relative to X

        Returns:
        ------- 
            Subgradient for w and w0.
        """       
        return (- self.C * np.sum(y),
                self.w - self.C * (X * y[:, np.newaxis]).sum(axis=0))

    def _is_stop(self, dQ_norm, dw_norm, t):
        """Checks stop criterion in order to break the loop.

        Parameters:
        ----------
        dQ_norm : decrement of loss function.
        dw_norm : norm of decrement of w
        t : current iteration
        """
        return (self.stop_criterion == 'objective') and (dQ_norm < self.tol) or \
               (self.stop_criterion == 'argument') and (dw_norm < self.tol) or \
               (t >= self.max_iter)

    def _fit_subgradient(self, X, y,
                         tol, max_iter, verbose,
                         stop_criterion, alpha, beta):
        """Subgradient Descent.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
            y : numpy.array, shape = [n_samples]
                Target vector relative to X
            tol : float, optional (default=1e-4)
              Tolerance for stopping criterion.

            max_iter : int
               The maximum number of iterations to be run.

            verbose : bool 
               Enable verbose output.

            stop_criterion : string
              The stop criteria. Must be 'objective' or 'argument'.
                -- 'objective' : the value of the objective function.
                -- 'argument'  : the value of the weights norm.
                
            alpha : float 
               Parameter for step in subgradient methods.

            beta : float 
               Parameter for step in subradient methods.

        Finds:
        -----
            w (weights): numpy.array, shape = [D]
                Solution of the primal SVM.

        Returns:
        -------
            The dictionary with fields:
            -- status : the reason for stopping the algorithm
                 0  - the stop criteria,
                 1  - the achievement of max_iter.
            -- objective_curve : list of values of the objective function.
            -- time : training time.
            """
        if self.method != 'subgradient':
            raise ValueError(
                  '_fit_subgradient method is not available. Please, check your method parameter')
            
        self.C = self._check_C(self.C)
        self.max_iter = self._check_max_iter(max_iter)
        self.alpha = float(self._check_step(alpha))
        self.stop_criterion = self._check_stop_criterion(stop_criterion)
        X, y = self._check_X_y(X, y)
        
        self.fit_method = self.method
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.beta = beta

        N, D = X.shape
        objective_curve = []

        self.w0 = 0
        self.w = np.zeros((D, ))

        n = np.arange(N)
        Q, ind = self.compute_primal_objective_ind(X, y, n)
        Q /= N
        t = 0
        if verbose:
            print('[subgradient {}], n_iter:{}, Q:{}'.format(stop_criterion, t, Q))
        objective_curve.append(Q) 
            
        start = time.clock()
        while True:
            t += 1
            
            sgQ_w0, sgQ_w = self.compute_subgradient(X[ind], y[ind])
            eta = alpha / t ** beta
            self.w = self.w - eta * sgQ_w
            self.w0 = self.w0 - eta * sgQ_w0
            
            Q_curr, ind = self.compute_primal_objective_ind(X, y, n)
            Q_curr /= N

            dw_norm = eta * np.linalg.norm(sgQ_w)  
            dQ_norm = abs(Q - Q_curr)
            
            if verbose:
                print('[subgradient {}], n_iter:{}, Q:{}'.format(stop_criterion, t, Q_curr))
            objective_curve.append(Q_curr) 
            
            if self._is_stop(dQ_norm, dw_norm, t):
                break
                
            Q = Q_curr  
            
        return {'status' : int(t == self.max_iter), 
                'objective_curve' : objective_curve, 
                'time' : time.clock() - start }

    def _fit_stoch_subgradient(self, X, y, 
                               tol, max_iter, verbose, 
                               stop_criterion, batch_size, 
                               lamb, alpha, beta, min_iter):
        """Stochastic Subgradient Descent.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
            y : numpy.array, shape = [n_samples]
                Target vector relative to X
                
            tol : float
              Tolerance for stopping criterion.

            max_iter : int
               The maximum number of iterations to be run.

            verbose : bool 
               Enable verbose output.

            stop_criterion : string
              The stop criteria. Must be 'objective' or 'argument'.
                -- 'objective' : the value of the objective function.
                -- 'argument'  : the value of the weights norm.

            batch_size : int
               Size of the batch for training in SGD.

            alpha : float
               Parameter for step in subgradient methods.

            beta : float
               Parameter for step in subradient methods.
               
            min_iter : int
               The minimum number of iterations to be run.

        Finds:
        -----
            w (weights): numpy.array, shape = [D]
                Solution of the primal SVM.

        Returns:
        -------
            The dictionary with fields:
            -- status : the reason for stopping the algorithm
                 0  - the stop criteria,
                 1  - the achievement of max_iter.
            -- objective_curve : list of values of the objective function.
            -- time : training time.
        """
        if self.method != 'stoch_subgradient':
            raise ValueError(
                  '_fit_stoch_subgradient method is not available. Please, check your method parameter') 
            
        self.C = self._check_C(self.C)
        self.max_iter = self._check_max_iter(max_iter)
        self.alpha = float(self._check_step(alpha))
        self.stop_criterion = self._check_stop_criterion(stop_criterion)
        X, y = self._check_X_y(X, y)
        
        self.fit_method = self.method
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.beta = beta

        if stop_criterion == 'objective':
            lamb = self._check_step(lamb)

        N, D = X.shape
        self.batch_size = self._check_batch_size(batch_size, N)
        objective_curve = []
        
        self.w0 = 0
        self.w = np.zeros((D, ))

        Q, ind = self.compute_primal_objective_ind(X, y, np.arange(N))
        Q /= N
        #ind = np.random.choice(N, batch_size, replace=False) 
        #_, ind = self.compute_primal_objective_ind(X[ind], y[ind])
        t = 0
        if verbose:
            print('[stoch_subgradient {}], n_iter:{}, Q:{}'.format(stop_criterion, t, Q))
        objective_curve.append(Q)
        
        start = time.clock()
        while True:
            t += 1
            
            sgQ_w0, sgQ_w = self.compute_subgradient(X[ind], y[ind])
            eta = alpha / t ** beta
            self.w = self.w - eta * sgQ_w
            self.w0 = self.w0 - eta * sgQ_w0
            
            ind = np.random.choice(N, batch_size, replace=False) 
            Q_curr, ind = self.compute_primal_objective_ind(X, y, ind) 
            Q_curr /= batch_size
            
            if stop_criterion == 'objective': 
                Q_curr = (1 - lamb) * Q + lamb * Q_curr
            else:
                Q_curr = (1 - 1.0 / t) * Q + Q_curr / t

            dw_norm = eta * np.linalg.norm(sgQ_w)
            dQ_norm = abs(Q - Q_curr)  
            
            if verbose:
                print('[stoch_subgradient {}], n_iter:{}, Q:{}'.format(stop_criterion, t, Q_curr))
            objective_curve.append(Q_curr) 
            
            if self._is_stop(dQ_norm, dw_norm, t) and t > min_iter:
                break
            Q = Q_curr
            
        return {'status' : int(t == self.max_iter), 
                'objective_curve' : objective_curve, 
                'time' : time.clock() - start }                        

    def _fit_liblinear(self, X, y, tol, max_iter, verbose):
        """Linear Support Vector Classification.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
                
            y : numpy.array, shape = [n_samples]
                Target vector relative to X
                
            tol : float
              Tolerance for stopping criterion.
              
            max_iter : int
               The maximum number of iterations to be run.
               
            verbose : bool 
               Enable verbose output.
        Finds:
        -----
            w (weights): numpy.array, shape = [D]
                Solution of the primal SVM.

        Returns:
        -------
            The dictionary with only one field:
            -- time : training time.
        """
        if self.method != 'liblinear':
            raise ValueError(
                  '_fit_liblinear method is not available. Please, check your method parameter')
        self.C = self._check_C(self.C)
        self.max_iter = self._check_max_iter(max_iter)
        X, y = self._check_X_y(X, y)
        
        self.fit_method = self.method

        lin_clf = svm.LinearSVC(dual=False, tol=tol, C=self.C,
                                    verbose=int(verbose), max_iter=max_iter)
        start = time.clock()
        lin_clf.fit(X, y) 
        self.w = lin_clf.coef_[0]
        self.w0 = lin_clf.intercept_[0]
        return {'time' : time.clock() - start}

    def _fit_libsvm(self, X, y, tol, max_iter, verbose):
        """Linear Support Vector Classification.

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.
                
            y : numpy.array, shape = [n_samples]
                Target vector relative to X
                
            tol : float
              Tolerance for stopping criterion.
              
            max_iter : int
               The maximum number of iterations to be run.
               
            verbose : bool
               Enable verbose output.

        Finds:
        -----
            A (values of the dual variables): numpy.array, shape = [N]
                Solution of the dual SVM. 

        Returns:
        -------
            The dictionary with only one field:
            -- time : training time.
        """
        if self.method != 'libsvm':
            raise ValueError(
                  '_fit_libsvm method is not available. Please, check your method parameter')
            
        self.C = self._check_C(self.C)
        self.max_iter = self._check_max_iter(max_iter)
        self.kernel = self._check_kernel(self.kernel)
        self.gamma = self._check_gamma(self.gamma, X.shape[1])
        X, y = self._check_X_y(X, y)
        
        self.fit_method = self.method
        self.tol = tol
        
        clf = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, 
                      verbose=int(verbose), tol=tol, max_iter=max_iter)
        start = time.clock()
        clf.fit(X, y) 
        self.A = np.zeros(X.shape[0])
        self.A[clf.support_] = np.abs(clf.dual_coef_[0])

        self.ind_sv = self.A > EPS
        self.sv = X[self.ind_sv]
        self.sv_y = y[self.ind_sv]

        return {'time' : time.clock() - start}

    def compute_support_vectors(self, X):
        """Finds support vectors. (For dual task only) 

        Parameters:
        ----------
            X : numpy.array, shape = [n_samples, n_features]
               Training vector, where n_samples in the number of samples and
               n_features is the number of features.   

        Returns:
        -------
            support_vectors_ : numpy.array, shape = [n_support_vectors, D] 
        """         
            
        if self.fit_method is None:
            raise NotFittedError("Must fit svm before querying.")

        if self.method in ('dual', 'libsvm'):
            return self.sv

        raise ValueError(
                    "Compute_support_vectors method is available only in the dual task")

    def compute_w(self, X=None, y=None):  
        """Finds primal variables of the dual.
        Available only in the dual linear task.

        Parameters:
        ----------
            X : optional (default=None), numpy.array, shape = [n_samples, n_features]
                Training vector, where n_samples in the number of samples and
                n_features is the number of features.

            y : optional (default=None), numpy.array, shape = [n_samples, 1]
                Target vector relative to X

        Returns:
        -------
            w : numpy.array, shape = [D, 1]
             Weights for primal task.
            """

        if self.fit_method is None:
            raise NotFittedError("Must fit svm before querying.")

        if self.kernel != 'linear':
            raise ValueError(
                    "Compute_w method is available only in the dual linear task")

        A_sv = self.A[self.ind_sv]
        w = np.sum(self.sv * (A_sv * self.sv_y)[:, np.newaxis], axis=0)

        ind_sv_edge = A_sv < self.C
        w0 = -np.sum(w * self.sv[ind_sv_edge][0]) + self.sv_y[ind_sv_edge][0] 

        return w.reshape(w.shape[0], 1)

    def predict(self, X_test, return_classes=False):
        """Perform classification on samples in X_test.
          
        Parameters:
        ----------
            X_test : numpy.array, shape (n_samples_test, n_features)

            return_classes : optional (defualt=False)
               if provided with True will return class labels,
               otherwise scores will be returned.

        Returns:
        -------
            y_pred : array, shape (n_samples, 1)
               Class labels for samples in X (or probabilities).
               Based on return_classes parameter.
        """
        if self.method in ('dual', 'libsvm'):
            return self.predict_dual(X_test, return_classes)

        if self.fit_method is None:
            raise NotFittedError("Must fit svm before querying.")

        X_test = self._check_array(X_test, atleast_2d=True)

        if X_test.shape[1] != self.w.shape[0]:
            raise ValueError("X has %d features per sample; expecting %d"
                              % (X_test.shape[1], self.w.shape[0]))

        f = np.dot(X_test, self.w) + self.w0

        return np.sign(f).astype(int) if return_classes else f

    def predict_dual(self, X_test, return_classes):
        """Perform classification on samples in X_test in dual task.

        Parameters:
        ----------
            X_test : numpy.array, shape (n_samples_test, n_features)

            return_classes : optional (defualt=False)
               if provided with True will return class labels,
               otherwise scores will be returned.

        Returns:
        -------
            y_pred : array, shape (n_samples, 1)
              Class labels for samples in X (or probabilities).
              Based on return_classes parameter.
        """ 
        if self.fit_method is None:
            raise NotFittedError("Must fit svm before querying.")

        X_test = self._check_array(X_test, atleast_2d=True)

        if X_test.shape[1] != self.sv.shape[1]:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X_test.shape[1], self.sv.shape[1]))

        if self.method not in ('dual', 'libsvm'):
            raise ValueError('_predict_dual method is not available. \
                                Please, check your method parameter')  

        A_sv = self.A[self.ind_sv]
        ind_sv_edge = A_sv < self.C + EPS
        
        if not ind_sv_edge.size:
            raise ValueError('Calculation error: please change C paramter')

        if self.kernel == 'linear':
            w = np.sum(self.sv * (A_sv * self.sv_y)[:, np.newaxis], axis=0)
            w0 = -np.sum(w * self.sv[ind_sv_edge][0]) + self.sv_y[ind_sv_edge][0]

            f = np.dot(X_test, w) + w0 
        else:
            K = self.rbf_kernel(self.sv, self.sv[ind_sv_edge][0], self.gamma)
            w0 = -np.sum(A_sv * self.sv_y * K) + self.sv_y[ind_sv_edge][0]

            K = self.rbf_kernel(self.sv, X_test, self.gamma)
            f = (K * (A_sv * self.sv_y)[:, np.newaxis]).sum(axis=0) + w0

        return np.sign(f).astype(int) if return_classes else f 

    def linear_kernel(self, X, Y):
        """Compute the linear kernel between X and Y.

        Parameters:
        ----------
            X : array of shape (n_samples_X, n_features)
            Y : array of shape (n_samples_Y, n_features)

        Returns:
        -------
            kernel_matrix : array of shape (n_samples_X, n_samples_Y)
        """
        X, Y = self._check_pairwise_arrays(X, Y)
        return np.dot(X, Y.T)

    def rbf_kernel(self, X, Y, gamma=None):
        """Compute the rbf (gaussian) kernel between X and Y::
        K(x, y) = exp(-gamma ||x-y||^2).
        For each pair of rows x in X and y in Y.

        Parameters:
        ----------
            X : array of shape (n_samples_X, n_features)
            Y : array of shape (n_samples_Y, n_features)
            gamma : float, default None
              If None, defaults to 1.0 / n_samples_X

        Returns:
        -------
            kernel_matrix : array of shape (n_samples_X, n_samples_Y)
        """
        X, Y = self._check_pairwise_arrays(X, Y)
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        K = euclidean_distances(X, Y, squared=True)
        K *= -gamma
        K = np.exp(K) 
        return K

    def _check_C(self, C):
        """Check to make sure C is valid."""
        if C <= 0:
            raise ValueError(
                "Penalty term must be positive; got (C=%r)" % C)
        return C
    
    def _check_max_iter(self, max_iter):
        """Check to make max_iter is valid."""
        if max_iter <= 0:
            raise ValueError(
                      "max_iter must be a positive; got (max_iter=%d)" % max_iter)
        return max_iter
    
    def _check_batch_size(self, batch_size, N):
        """Check to make sure batch_size is valid."""

        if batch_size <= 0 or batch_size > N:
            raise ValueError(
                    "batch_size must be in (0, %d]; got (batch_size=%d)" %
                     (N, batch_size) )
        return batch_size
          
    def _check_step(self, h):
        """Check to make sure step is valid."""

        if h <= 0:
            raise ValueError(
                    "h must be positive; got (h=%r)" %
                     h)
        return h

    def _check_stop_criterion(self, stop_criterion):
        """Check to make sure stop_criterion is valid."""

        if stop_criterion in ('objective', 'argument'):
            return stop_criterion
        else:
            raise ValueError(
                "stop_criterion not recognized: should be 'objective' or 'argument'")


    def _check_kernel(self, kernel):
        """Check to make sure kernel is valid."""

        if kernel in ('linear', 'rbf'):
            return kernel

        raise ValueError("kernel must be set up: should be 'linear' or 'rbf'" )
        
    def _check_gamma(self, gamma, D):
        """Check to make sure gamma is valid."""
        if gamma is None:
            return 1.0 / D
        
        if gamma < 0:
            raise ValueError(
                    "gamma must be positive; got (gamma=%r)" %
                     gamma)

        return gamma

    def _check_method(self, method):
        """Check to make sure method is valid"""

        if method in (
                'primal',
                'dual',
                'subgradient',
                'stoch_subgradient',
                'liblinear',
                'libsvm'):
                return method
        else:
            raise ValueError(
                    "method not recognized: should be 'primal', 'dual', 'subgradient', 'stoch_subgradient', 'liblinear', 'libsvm'")

    def _check_X_y(self, X, y, ensure_2d=True):
        """Checks X and y for consistent length, enforces X 2d and y 1d.

        Parameters:
        ----------
            X : numpy.array
              Input data.
            y : numpy.array
              Labels.

            ensure_2d : boolean (default=True)
              Whether to make X at least 2d.

        Returns:
        -------
            X_converted : numpy.array
              The converted and validated X.
            y_converted : numpy.array
              The converted and validated y.
        """
        X = self._check_array(X, atleast_2d=True)
        y = self._check_array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                    "Found input variables with inconsistent number of"
                    " samples: %r" %
                    X.shape[0])

        return X, y.reshape(y.shape[0], )

    def _check_array(self, array, atleast_2d=False):
        """Input validation on an array, list

        Parameters:
        ----------
            array : array-like
              Input object to check / convert.
            ensure_2d : boolean, optional (default=False)
              Whether to make X at least 2d.
        """

        if not isinstance(array, np.ndarray):
            raise ValueError("Incompatible type of vector. Must be numpy.array")

        if atleast_2d and array.ndim < 2:
            array = np.atleast_2d(array)

        return array

    def _check_pairwise_arrays(self, X, Y):
        """This function first ensures that both X and Y are arrays,\
           and  that the size of the second dimension of the two arrays is equal.
        """

        X = self._check_array(X, atleast_2d=True)
        Y = self._check_array(Y, atleast_2d=True)

        if X.shape[1] != Y.shape[1]:
            raise ValueError("Incompatible dimension for X and Y matrices: "
                             "X.shape[1] == {}\
                              while Y.shape[1] == {}".format(X.shape[1],
                                                        Y.shape[1]
                                                        )
                             )
        return X, Y

class NotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting.
       This class inherits from ValueError"""