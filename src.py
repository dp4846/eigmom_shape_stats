import numpy as np
import cvxopt as opt
from scipy.stats import multivariate_normal as mvn
from math import comb

def estimate_eigenmoments(X, Y, num_moments, comb_M_2ps=None):
    """
    Compute unbiased moment estimates.

    Parameters
    ----------
    X : ndarray
        (samples x neurons) matrix of responses.
    Y : ndarray
        (samples x neurons) matrix of responses.
    num_moments : int
        Maximum moment to calculate.

    Returns
    -------
    eigmoms : ndarray
        Estimates for first P moments (including zero).
    """

    # TODO: allow X and Y to have different # neurons.
    #assert Y.shape == X.shape
    M, N_x = X.shape
    M, N_y = Y.shape
    N = min([N_x, N_y])#number of singular values is min of # neurons in X and Y

    if comb_M_2ps is None:
        # Pre-compute combinatorial terms.
        comb_M_2ps = np.array([comb(M, 2 * p) for p in range(1, num_moments + 1)])

    # Pre-compute covariances.
    XXt = X @ X.T / M
    YYt = Y @ Y.T / M
    eigmoms = np.zeros(num_moments + 1)
    eigmoms[0] = N 

    # Precompute matrix products.
    G0 = np.triu(XXt, 1) @ np.triu(YYt, 1)
    G = np.triu(XXt, 1) @ YYt

    # Compute first eigenmoment
    eigmoms[1] = (M**2) * np.trace(G) / comb_M_2ps[0]

    for p in range(2, num_moments + 1):
        #Compute matrix power iteratively.
        G = G0 @ G
        #Compute p-th eigenmoment
        #the first factor inverts normalization by M
        eigmoms[p] = M**(2 * p) * np.trace(G) / comb_M_2ps[p - 1]

    return eigmoms

def bootstrap_eigmoms_cov(X, Y, num_moments,num_bootstraps=50):
    """
    Compute covariance of eigenmoments using bootstrapping.

    Parameters
    ----------
    X : ndarray
        (samples x neurons) matrix of responses.
    Y : ndarray
        (samples x neurons) matrix of responses.
    num_moments : int
        Maximum moment to calculate.
    num_bootstraps : int
        Number of bootstraps to use to estimate covariance.
    
    Returns
    -------
    cov : ndarray
        Bootstrap covariance matrix of eigenmoments.
    """
    M, _ = X.shape
    boot_eigmoms = np.empty(
        (num_moments + 1, num_bootstraps)
    )
    if comb_M_2ps is None:
        # Pre-compute combinatorial terms.
        comb_M_2ps = np.array([comb(M, 2 * p) for p in range(1, num_moments + 1)])

    for i in (range(num_bootstraps)):
        inds = np.random.choice(np.arange(M), size=M, replace=True)#sample with replacement from stimuli/samples
        boot_eigmoms[:, i] = estimate_eigenmoments(
            X[inds], Y[inds], num_moments, comb_M_2ps=comb_M_2ps
        )
    return np.cov(boot_eigmoms)

def calc_gammas(
        X, Y, num_moments,
        n_domain_pts=500,
        num_bootstraps=50,
        alpha=1.0,
        eig_mom_cov=None,
        max_eig=None,
        remove_constant=False,
        return_poly_basis=False,
        bias_frac=0.05,#what fraction, relative to upperlimit on nuclear norm
        denom = None,
    ):
    """
    Compute the weights (gammas) on eigenmoments to estimate f(\lambda)
    Parameters
    ----------
    X : ndarray
        (samples x neurons) matrix of responses.
    Y : ndarray
        (samples x neurons) matrix of responses.
    num_moments : int
        Maximum eigenmoment moment to calculate.
    n_domain_pts : int
        Number of points to approximate f() at.
    num_bootstraps : int
        Number of bootstraps to use to estimate covariance.
    alpha : float
        Regularization parameter (0 minimizes variance, 1 minimize MSE, 2 minimize bias).
    eig_mom_cov : ndarray
        Covariance matrix of eigenmoments if already computed.
    max_eig : float
        Maximum eigenvalue of E[X.T @ Y / M], unknown usually but need to estimate to determine limit
        of approximation domain
    remove_constant : bool
        Remove constant term from polynomial basis, this ensures estimator is not biased at 0.
    return_poly_basis : bool
        Return the polynomial basis used to compute the estimator for plotting.
    bias_frac : float
        Fraction of upper limit on bias to use as constraint.
    denom : float
        Denominator of estimator, if None then computed from X and Y.
    Returns
    -------
    gammas : ndarray
        Weights on eigenmoments.
    poly_basis : ndarray
        Polynomial basis used to compute estimator.
    """

    M, N_x = X.shape #TODO no need to pass whole variables, just pass M and N
    M, N_y = Y.shape
    N = min([N_x, N_y])#number of singular values is min of # neurons in X and Y
    if eig_mom_cov is None:
        eig_mom_cov = bootstrap_eigmoms_cov(X, Y, num_moments, num_bootstraps)
    if denom is None:
        denom = (np.trace(X.T@X/M)*np.trace(Y.T@Y/M))**0.5
    upper_bias_lim = bias_frac*denom #set the limit on bias, comes at expense of variance
    if max_eig is None:
        #max_eig = (np.linalg.norm(X.T @ Y, ord=2) / M) ** 2
        max_eig = np.linalg.svd(X.T @ Y / M)[1][0] ** 2 #get first singular value squared of cross-covariance as upper lim

    # Objective, x^T P x + q^T x
    P = np.zeros((num_moments + 2, num_moments + 2))
    P[:(num_moments+1), :(num_moments+1)] = (2 - alpha) * eig_mom_cov
    P[-1, -1] = alpha * (N ** 2)
    #P += np.eye(num_moments + 2)
    q = np.zeros(num_moments + 2)

    # Constraints, Gx <= h

    xt = np.linspace(0, max_eig, n_domain_pts)
    M = np.array([xt**i for i in range(0, num_moments + 1)]).T
    h = np.array(list(np.sqrt(xt)) +  list(-np.sqrt(xt)) + [upper_bias_lim, upper_bias_lim])
    G = np.concatenate(
    [np.concatenate([M, -np.ones((M.shape[0], 1))], 1),#error between true function and moments as a function of x
        np.concatenate([-M, -np.ones((M.shape[0], 1))], 1),
        np.array([0,]*(M.shape[1]) + [N,])[None],
        np.array([0,]*(M.shape[1]) + [-N,])[None]], 0 #constrain max deviations to be less than bias
        )
    if remove_constant:# so that constant part of polynomial isn't used so that 0 is always 0
        P = P[1:, 1:]
        q = q[1:]
        G = G[:, 1:]
    # Optimize.
    P = opt.matrix(P)
    q = opt.matrix(q)
    G = opt.matrix(G) # constraints
    h = opt.matrix(h)
    sol = opt.solvers.qp(
        P=P, q=q, G=G, h=h,
        options={'show_progress': False, },
        solver='OSQP'
    )
    gamma = np.array(sol['x'][:-1]).ravel()
    if remove_constant:
        gamma = np.concatenate([[0,], gamma])# so that it works with all estimated moments
    min_max_abs_dev = sol['x'][-1] * N
    if return_poly_basis:
        return gamma, min_max_abs_dev, M, xt
    else:
        return gamma, min_max_abs_dev

def full_similarity_metric_estimator(X, Y, num_moments=10, alpha=1, 
                                            num_bootstraps=500, 
                                            remove_constant=True,
                                            bias_frac=1, comb_M_2ps=None):
    """
    Complete function to estimate similarity metric between two noisy vectors of signals
    Parameters
    ----------
    X : ndarray
        (samples x neurons) matrix of responses.
    Y : ndarray
        (samples x neurons) matrix of responses.
    num_moments : int
        Maximum eigenmoment moment to calculate.
    alpha : float
        Regularization parameter (0 minimizes variance, 1 minimize MSE, 2 minimize bias).
    num_bootstraps : int
        Number of bootstraps to use to estimate covariance.
    remove_constant : bool
        Remove constant term from polynomial basis, this ensures estimator is not biased at 0.
    bias_frac : float
        Fraction of upper limit on bias to use as constraint.
    Returns
    -------
    est : float
        Estimated similarity metric.
    var : float
        Variance of estimator.
    bias : float
        Bias of estimator.
    naive_est : float
        Naive estimator (just uses nuclear norm of sample cross-covariance).
    """

    M = X.shape[0] #TODO no need to pass whole variables, just pass M and N
    M = Y.shape[0]
    cov_est = (X.T @ Y)/(M)#TODO extend to more than two repeats
    s1_est = (np.linalg.svd(cov_est)[1][0])#TODO no need to do this twice, could just scale estimate
    X /= s1_est**0.5
    Y /= s1_est**0.5
    cov_est = (X.T @ Y)/(M)
    naive_est_nuc_norm  = np.linalg.norm(cov_est, ord='nuc')
    denom_est = (np.trace(X.T@X/M)*np.trace(Y.T@Y/M))**0.5 
    if comb_M_2ps is None:
        comb_M_2ps = np.array([comb(M, 2 * p) for p in range(1, num_moments + 1)])

    est_eigmoms = estimate_eigenmoments(X, Y, num_moments, comb_M_2ps=comb_M_2ps)
    eig_mom_cov = bootstrap_eigmoms_cov(X, Y, num_moments, num_bootstraps=num_bootstraps, comb_M_2ps=comb_M_2ps) 
    
    gamma, bias = calc_gammas(
        X, Y, num_moments,
        alpha=alpha,
        eig_mom_cov=eig_mom_cov,
        max_eig=1, #TODO calculate max eig as opposed to relying on normalizing data beforehand
        remove_constant=remove_constant, 
        return_poly_basis=False,
        bias_frac=bias_frac,
        denom=denom_est)

    var = gamma@eig_mom_cov@gamma/(denom_est**2)#estimate of variance of similarity metric
    est =  est_eigmoms @ gamma/denom_est#estimate of similarity metric
    naive_est = naive_est_nuc_norm/denom_est#naive estimate of similarity metric using nuclear norm of sample cros-cov
    bias = bias/denom_est#estimate of bias of similarity metric
    return est, var, bias, naive_est

#make estimator 
def full_signal_similarity_metric_estimator(X, Y, num_moments=10, alpha=1, 
                                            num_bootstraps=500, remove_constant=True,
                                                            bias_frac=1):
    """
    Complete function to estimate similarity metric between two noisy vectors of signals
    Parameters
    ----------
    X : ndarray
        (repeats x samples x neurons) matrix of responses.
    Y : ndarray
        (repeats x samples x neurons) matrix of responses.
    num_moments : int
        Maximum eigenmoment moment to calculate.
    alpha : float
        Regularization parameter (0 minimizes variance, 1 minimize MSE, 2 minimize bias).
    num_bootstraps : int
        Number of bootstraps to use to estimate covariance.
    remove_constant : bool
        Remove constant term from polynomial basis, this ensures estimator is not biased at 0.
    bias_frac : float
        Fraction of upper limit on bias to use as constraint.
    Returns
    -------
    est : float
        Estimated similarity metric.
    var : float
        Variance of estimator.
    bias : float
        Bias of estimator.
    naive_est : float
        Naive estimator (just uses nuclear norm of sample cross-covariance).
    """

    _, M, _ = X.shape[1] #TODO no need to pass whole variables, just pass M and N
    _, M, _ = Y.shape[1]
    signal_cov_est = (X[0].T @ Y[1] + X[1].T @ Y[0])/(2*M)#TODO extend to more than two repeats
    s1_est = (np.linalg.svd(signal_cov_est)[1][0])#TODO no need to do this twice, could just scale estimate
    X /= s1_est**0.5
    Y /= s1_est**0.5
    signal_cov_est = (X[0].T @ Y[1] + X[1].T @ Y[0])/(2*M)
    naive_est_nuc_norm  = np.linalg.norm(signal_cov_est, ord='nuc')
    denom_est = (np.trace(X[0].T@X[1]/M)*np.trace(Y[0].T@Y[1]/M))**0.5 

    est_eigmoms = (estimate_eigenmoments(X[0], Y[1], num_moments) + estimate_eigenmoments(X[1], Y[0], num_moments))/2
    eig_mom_cov = (bootstrap_eigmoms_cov(X[0], Y[1], num_moments, num_bootstraps=num_bootstraps) 
                    + bootstrap_eigmoms_cov(X[1], Y[0], num_moments, num_bootstraps=num_bootstraps))/4
    
    gamma, bias = calc_gammas(
        X[0], Y[0], num_moments,
        alpha=alpha,
        eig_mom_cov=eig_mom_cov,
        max_eig=1, #TODO calculate max eig as opposed to relying on normalizing data beforehand
        remove_constant=remove_constant, 
        return_poly_basis=False,
        bias_frac=bias_frac,
        denom=denom_est)

    var = gamma@eig_mom_cov@gamma/(denom_est**2)#estimate of variance of similarity metric
    est =  est_eigmoms @ gamma/denom_est#estimate of similarity metric
    naive_est = naive_est_nuc_norm/denom_est#naive estimate of similarity metric using nuclear norm of sample cros-cov
    bias = bias/denom_est#estimate of bias of similarity metric
    return est, var, bias, naive_est

def sim_data(N_x, N_y, shape_metric, pl_slope, M, n_sims):
    """
    Generate simulated data for testing similarity metric estimator (no noise)
    Parameters
    ----------
    N_x : int
        Number of samples in X.
    N_y : int
        Number of samples in Y.
    shape_metric : float
        Desired shape metric value.
    pl_slope : float
        Power law slope of eigenvalue spectrum (0 highest dimensionality, >0 lower dimensionality).
    M : int
        Number of neurons.
    n_sims : int
        Number of simulations to run.
    Returns
    -------
    X : ndarray
        (simulation X  samples x neurons) matrix of responses.
    Y : ndarray
        (simulation X samples x neurons) matrix of responses.
    """

    max_N = np.max([N_x, N_y])
    min_N = np.min([N_x, N_y])
    inds = np.arange(1, max_N+1)
    spectrum = inds**(-pl_slope)
    singular_values = spectrum[:min_N]
    lam_x = spectrum[:N_x]
    lam_y = spectrum[:N_y]

    #set to desired shape metric value
    true_denom = (np.sum(lam_x)*np.sum(lam_y))**0.5
    true_num = np.sum(singular_values)
    singular_values = singular_values * (shape_metric*true_denom)/true_num
    true_num = np.sum(singular_values)
    cc = np.zeros((N_x, N_y))
    cc[np.diag_indices(N_x)] = singular_values
    Sigma_x = np.diag(lam_x)
    Sigma_y = np.diag(lam_y)
    cov = np.block([[Sigma_x, cc],
                    [cc.T, Sigma_y]])

    #now randomly rotate with orthogonal matrix on block diagonal
    rot_X = np.linalg.svd(np.random.randn(N_x, N_x))[0].T
    rot_Y = np.linalg.svd(np.random.randn(N_y, N_y))[0].T
    rot = np.block([[rot_X, np.zeros((N_x, N_y))],
                    [np.zeros((N_y, N_x)), rot_Y]])

    cov = rot @ cov @ rot.T
    R = mvn.rvs(mean=np.zeros(N_x+N_y), cov=cov, size=(n_sims, M))
    X = R[..., :N_x]
    Y = R[..., N_x:]
    return X, Y, singular_values, lam_x, lam_y, cov