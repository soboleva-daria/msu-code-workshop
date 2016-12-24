import numpy as np
import warnings

def check_w(w, n_components):
    """Check the provided weights."""

    w = check_array(w)
    check_shape(w, (n_components, ), 'w')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'w' should be in the range "
                         "[0, 1], but got max value {}, min value {}"
                         .format(np.min(w), np.max(w)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = {}".format(np.sum(weights)))
    return w


def check_mu(mu, n_components, n_features):
    """Validate the provided means."""

    mu = check_array(mu)
    check_shape(mu, (n_components, n_features), 'mu')
    return mu


def check_sigma(sigma, diag, n_components, n_features):
    """Validate the provided sigma."""

    sigma = check_array(sigma)
    if diag:
        check_shape(sigma, (n_components, n_features), 'sigma')
    else:
        check_shape(sigma, (n_components, n_features, n_features), 'sigma')
    return sigma


def check_X(X, n_components):
    """Validate the provided X."""

    X = check_array(X, atleast_2d=True)

    if X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = {}, n_samples = {}'
                         .format((n_components, X.shape[0])))
    return X


def check_shape(array, array_shape, name):
    """Validate the shape of the input parameter array."""

    if array.shape != array_shape:
        raise ValueError(
            "The parameter {} should have the shape of {}, "
            "but got {}".format(
                name, param_shape, param.shape))


def check_array(array, atleast_2d=False):
    """Input validation on an numpy.ndarray"""

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if atleast_2d and array.ndim < 2:
        array = np.atleast_2d(array)
    return array


def initialize_parameters(X, n_components, reg_cov, diag):
    """Initialize the model parameters.
    Parameters:
    ==========
    X : numpy.ndarray, shape  (n_samples, n_features)

    n_components : int,
        The number of mixture components.

    reg_cov : float,
        Non-negative regularization added to the diagonal of covariance.
         Allows to assure that the covariance matrices are all positive.

    diag : bool,
       If provided with True, each component of the mixture
       has its own diagonal covariance matrix, otherwise
       each component has its own general covariance matrix.
       
    Returns:
    ========
    w_start : numpy.ndarray, shape (n_components,)
        The numbers of data samples in the current components.

    mu_start : numpy.ndarray, shape (n_components, n_features)
        The centers of the current components.

    sigma_start : numpy.ndarray,
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    n_samples = X.shape[0]
    resp = np.random.rand(n_samples, n_components)
    resp /= resp.sum(axis=1)[:, np.newaxis]

    w_start, mu_start, sigma_start = estimate_gaussian_parameters(
        X, resp, reg_cov, diag)
    w_start /= n_samples
    return w_start, mu_start, sigma_start


def estimate_gaussian_parameters(X, resp, reg_cov, diag):
    """Estimate the Gaussian distribution parameters.

    Parameters:
    ===========
    X : numpy.ndarray, shape  (n_samples, n_features)

    resp : numpy.ndarray, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_cov : float
        The regularization added to the diagonal of the covariance matrices.

    diag : bool,
       If provided with True, each component of the mixture
       has its own diagonal covariance matrix, otherwise
       each component has its own general covariance matrix.

    Returns:
    ========
    nk : numpy.ndarray, shape (n_components,)
        The numbers of data samples in the current components.

    mu : numpy.ndarray, shape (n_components, n_features)
        The centers of the current components.

    sigma : numpy.ndarray,
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + np.finfo(resp.dtype).eps
    mu = np.dot(resp.T, X) / nk[:, np.newaxis]

    if diag:
        sigma = estimate_gaussian_sigma_diag(X, resp, nk, mu, reg_cov)
    else:
        sigma = estimate_gaussian_sigma(X, resp, nk, mu, reg_cov)
    
    return nk, mu, sigma


def estimate_gaussian_sigma_diag(X, resp, nk, mu, reg_cov):
    """Estimate the diagonal covariance vectors.

    Parameters:
    ===========
    X : numpy.ndarray, shape (n_samples, n_features)

    resp : numpy.ndarray, shape (n_samples, n_components)
      The responsibilities for each data sample in X.

    nk : numpy.ndarray, shape (n_components,)
      The numbers of data samples in the current components.

    mu : numpy.ndarray, shape (n_components, n_features)
      The centers of the current components.

    reg_cov : float
       The regularization added to the diagonal of the covariance matrices.

    Returns:
    ========
    sigma : numpy.ndarray, shape (n_components, n_features)
        The covariance vector of the current components.
    """

    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_mu2 = mu ** 2
    avg_X_mu = mu * np.dot(resp.T, X) / nk[:, np.newaxis]

    return avg_X2 - 2 * avg_X_mu + avg_mu2 + reg_cov


def estimate_gaussian_sigma(X, resp, nk, mu, reg_cov):
    """Estimate the general covariance matrices.

    Parameters:
    ===========
    X : numpy.ndarray, shape (n_samples, n_features)

    resp : numpy.ndarray, shape (n_samples, n_components)
      The responsibilities for each data sample in X.

    nk : numpy.ndarray, shape (n_components,)
      The numbers of data samples in the current components.

    mu : numpy.ndarray, shape (n_components, n_features)
      The centers of the current components.

    reg_cov : float
       The regularization added to the diagonal of the covariance matrices.

    Returns:
    ========
    sigma : numpy.ndarray, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = mu.shape

    sigma = np.empty((n_components, n_features, n_features))

    for k in range(n_components):
        diff = X - mu[k]
        sigma[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        sigma[k].flat[::n_features + 1] += reg_cov

    return sigma


def estimate_log_prob_resp(X, w, mu, sigma, diag):
    """Estimate log probabilities and responsibilities for each sample.

    Parameters:
    ==========
    X : numpy.ndarray, shape (n_samples, n_features)

    w : numpy.ndarray, shape (1, n_components)
        The weights of each mixture components.

    mu : numpy.ndarray, shape (n_components, n_features)
      The centers of the current components.

    sigma : numpy.ndarray,
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    Returns:
    ========
    log_prob_norm : numpy.array, shape (n_samples,)
        log p(X)
    log_resp : numpy.ndarray, shape (n_samples, n_components)
        logarithm of the responsibilities
    """
    mixture_prob = []
    if diag:
        for mu_k, sigma_k in zip(mu, sigma):
            mixture_prob.append(estimate_log_gaussian_prob(X, mu_k, np.diag(sigma_k)))
    else:
        for mu_k, sigma_k in zip(mu, sigma):
            mixture_prob.append(estimate_log_gaussian_prob(X, mu_k, sigma_k))
            
    # log P(X | Z) + log weights
    weighted_log_prob = np.array(mixture_prob) + np.log(w)[:, np.newaxis]
    
    log_prob_norm = logsumexp(weighted_log_prob)
    log_resp = weighted_log_prob - log_prob_norm

    return log_prob_norm, log_resp.T


def logsumexp(arr):
    """Returns log(sum(exp(arr))). Use the max to normalize"""
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out

def estimate_log_gaussian_prob(X, mu, sigma):
    """Estimate log gaussian probabilities.

    Parameters:
    ==========
    X : numpy.ndarray, shape (n_samples, n_features)

    mu : numpy.ndarray, shape (n_components, n_features)
      The centers of the current components.

    sigma : numpy.ndarray,
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """

    N = X.shape[0]
    D = X.shape[1]

    inv = np.linalg.inv(sigma)
    _, logdet = np.linalg.slogdet(sigma)
    const = D * np.log(2 * np.pi) + logdet
    X_mu = X - mu
    right_part = np.dot(X_mu, inv.T)
    return -0.5 * (const + (X_mu * right_part).sum(axis=1))


def run_em(
        X,
        n_components=1,
        tol=1e-03,
        reg_cov=1e-06,
        max_iter=100,
        diag=True,
        w_start=None,
        mu_start=None,
        sigma_start=None):
    """run_em method implements the expectation-maximization (EM) algorithm
    for fitting mixture-of-Gaussian models.

    Paramters:
    ==========

    X : numpy.ndarray, shape (n_samples, n_features)

    n_components : int, optional, defaults to 1.
        The number of mixture components.

    tol : float, optional, defaults to 1e-3.
      The convergence threshold. EM iterations will stop when the
      lower bound average gain is below this threshold.

    reg_cov : float, defaults to 1e-06.
        Non-negative regularization added to the diagonal of covariance.
         Allows to assure that the covariance matrices are all positive.

    max_iter : int, optional, defaults to 100.
        The number of EM iterations to perform.

    diag : bool, optional, defualts True
       If provided with True, each component of the mixture
       has its own diagonal covariance matrix, otherwise
       each component has its own general covariance matrix.

    w_start : numpy.ndarray, shape (n_components, ), optional
        Initial weights, defaults to None.
        If it None, weights are initialized using random initialization.

    mu_start: numpy.ndarray, shape (n_components, n_features),
              optional, defualts None
        Initial means, defaults to None,
        If it None, means are initialized using random initialization.

    sigma_start: numpy.ndarray, optional.
        Initial matrices of covariance, defaults to None.
        If it None, matrices of covariance are initialized using random initialization.
        The shape depends on 'covariance_type'::
            (n_components, n_features)             if diag == True,
            (n_components, n_features, n_features) if diag == False.


    Returns:
    ========
    Tuple of elements:

    w : numpy.ndarray, shape (1, n_components)
        The weights of each mixture components.

    mu : numpy.ndarray, shape (n_components, n_features)
        The mean of each mixture component.

    sigma : numpy.ndarray.
        The covariance of each mixture component.
        The shape depends on `covariance_type`::
            (n_components, n_features)             if diag == True,
            (n_components, n_features, n_features) if diag == False.

    log_likelyhood_curve :
     List of values of the log-likelihood on each iteration.
    """
    if n_components < 1:
        raise ValueError("Invalid value for 'n_components': {} "
                         "Estimation requires at least one component"
                         .format(n_components))

    if tol < 0.:
        raise ValueError("Invalid value for 'tol': {} "
                         "Tolerance used by the EM must be non-negative"
                         .format(tol))

    if max_iter < 1:
        raise ValueError("Invalid value for 'max_iter': {} "
                         "Estimation requires at least one iteration"
                         .format(max_iter))
        
    X = check_X(X, n_components)
    
    w, mu, sigma = initialize_parameters(X, n_components, reg_cov, diag)
    
    if w_start is not None:
        w = check_w(w_start, n_components)

    if mu_start is not None:
        mu = check_mu(mu_start, n_components, n_features)

    if sigma_start is not None:
        sigma = check_sigma(
            sigma_start,
            n_components,
            n_features,
            diag)
    else:
        diag = True
        
    lower_bound = -np.inf
    log_likelyhood_curve = []
    
    for n_iter in range(max_iter):

        # E-step
        log_prob_norm, log_resp = e_step(X, w, mu, sigma, diag)
        log_likelyhood_curve.append(log_prob_norm)

        # M-step
        w, mu, sigma = m_step(X, log_resp, reg_cov, diag)

        change = log_prob_norm - lower_bound
        if abs(change) < tol:
            break

    return (w.reshape(1, n_components), mu, sigma, log_likelyhood_curve)


def e_step(X, w, mu, sigma, diag):
    """E step.
    Parameters:
    ===========
    X : numpy.ndarray, shape (n_samples, n_features)

    w : numpy.ndarray, shape (1, n_components)
        The weights of each mixture components.

    mu : numpy.ndarray, shape (n_components, n_features)
      The centers of the current components.

    sigma : numpy.ndarray,
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    Returns:
    ========
    log_prob_norm : float
        Sum of the logarithms of the probabilities of each sample in X

    log_resp : numpy.ndarray, shape (n_samples, n_components)
        Logarithm of the posterior probabilities (or responsibilities) of
        the point of each sample in X.
    """
    log_prob_norm, log_resp = estimate_log_prob_resp(X, w, mu, sigma, diag)
    return np.sum(log_prob_norm), log_resp


def m_step(X, log_resp, reg_cov, diag):
    """M step.

    Parameters:
    ==========
    X : numpy.ndarray, shape (n_samples, n_features)

    log_resp : numpy.ndarray, shape (n_samples, n_components)
        logarithm of the responsibilities.

    Return:
    =======
    w : numpy.ndarray, shape (1, n_components)
     The weights of each mixture components.

    mu : numpy.ndarray, shape (n_components, n_features)
     The centers of the current components.

    sigma : numpy.ndarray,
       The covariance matrix of the current components.
       The shape depends of the covariance_type.

    """
    w, mu, sigma = estimate_gaussian_parameters(
        X, np.exp(log_resp), reg_cov, diag)
    
    w /= X.shape[0]


    return w, mu, sigma

def run_em_with_restarts(
        X,
        n_restarts=1,
        n_components=1,
        tol=1e-03,
        reg_cov=1e-06, 
        max_iter=100,
        diag=True):

    """Please refer to run_em method. The only difference is that\
       run_em_with_restarts will use different initializations.

    Paramters:
    ==========

    n_restarts : int, optional (default 1)
        The number of initializations to perform. The best results are kept.

    Returns:
    ========
    Tuple of elements from run_em for best_initialization.
    """
    best_log_lh = -np.inf

    for init in range(n_restarts):
        w, mu, sigma, log_lh_curve = run_em(
            X, n_components, tol, reg_cov, max_iter, diag)
      
        max_log_lh = max(log_lh_curve)
        if max_log_lh > best_log_lh:
            best_log_lh = max_log_lh 
            best_log_lh_curve = log_lh_curve
            best_w = w
            best_mu = mu
            best_sigma = sigma

    if len(best_log_lh_curve)  == max_iter:
        warnings.warn('Initialization {} did not converged. '
                      'Try different init parameters, '
                      'or increase max_iter, tol'.format(init + 1))
            
    return (best_w, best_mu, best_sigma, best_log_lh_curve)