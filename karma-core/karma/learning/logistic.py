import numpy as np
import numexpr as ne
from cyperf.matrix.karma_sparse import is_karmasparse
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model.logistic import LogisticRegression

from karma.core.utils import create_timer
from karma.learning.matrix_utils import diagonal_of_inverse_symposdef
from karma.learning.regression import regression_on_blocks
from karma.learning.utils import VirtualHStack
from karma.thread_setter import blas_threads, open_mp_threads
from karma.runtime import KarmaSetup


__all__ = ['logistic_coefficients']


# TODO : I want to have a context for this
ne.set_num_threads(4)


def expit(x):
    """
    # It about x3 faster with numexpr
    # It's equivalent to "1/(1 + exp(-x))", but more robust for larger values

    >>> from scipy.special import expit as sy_expit
    >>> x = np.random.randn(10**4) * 30
    >>> np.allclose(sy_expit(x), expit(x))
    True
    """
    return ne.evaluate('(tanh(x / 2) + 1) / 2', truediv=True)


def _expit_m1_times_y(x, y):
    return ne.evaluate('y * (tanh(x/2) - 1) / 2', truediv=True)


def _log_logistic_inplace(x):
    return ne.evaluate('where(x > 0, -log(1 + exp(-x)), x - log(1 + exp(x)))', out=x)


def _log_logistic_inplace_sample_weight(x, sw):
    return ne.evaluate('where(x > 0, - sw * log(1 + exp(-x)), sw * (x - log(1 + exp(x))))',
                       out=x, casting='same_kind')


def logistic_loss_and_grad(w, X, y, alpha, sample_weight=None,
                           w0=None, alpha_intercept=0., intercept0=0.):

    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    fit_intercept = (w.size == n_features + 1)
    c, w = (w[-1], w[:-1]) if fit_intercept else (0, w)

    if w0 is None:
        w0 = np.zeros_like(w)

    # Downcast to have faster expit and _log_logistic_inplace
    yz = X.dot(w).astype(np.float32, copy=False)
    yz += c
    yz *= y
    z0 = _expit_m1_times_y(yz, y)

    dw = alpha * (w - w0)
    out = .5 * dw.dot(w - w0)

    if sample_weight is None:
        out -= _log_logistic_inplace(yz).sum()
    else:
        out -= _log_logistic_inplace_sample_weight(yz, sample_weight).sum()
        z0 *= sample_weight

    grad[:n_features] = X.transpose_dot(z0)
    grad[:n_features] += dw

    if fit_intercept:
        grad[-1] = z0.sum() + alpha_intercept * (c - intercept0)
    return out, grad


def logistic_coefficients_lbfgs(X, y, max_iter, C=1e10, w_warm=None, sample_weight=None, nb_threads=1):
    """
    >>> from karma.core.dataframe import DataFrame
    >>> from karma.core.utils.utils import use_seed
    >>> with use_seed(123):
    ...     df = DataFrame({'a': np.random.poisson(15, size=(1000, 10)), 'b': np.random.poisson(15, size=(1000, 3)),
    ...                     'c': np.random.poisson(15, size=(1000, 14)), 'y': np.random.randint(0, 2, 1000)})
    >>> with use_seed(1237):
    ...     res1 = df.build_lib_column('logistic_regression', ('a', 'b', 'c'),
    ...                 {'solver': 'lbfgs', 'axis': 'y', 'C': np.array([10.] * 10 + [1000.] * 3 + [10000000.] * 14)})
    >>> with use_seed(1237):
    ...     res2 = df.build_lib_column('logistic_regression', ('a', 'b', 'c'),
    ...                 {'solver': 'lbfgs', 'axis': 'y', 'C': [10., 1000., 10000000.]})
    >>> np.array_equal(res1[:], res2[:])
    True
    >>> df['0'] = df.build_lib_column('logistic_regression', ('a', 'b', 'c'),
    ...                 {'solver': 'lbfgs', 'axis': 'y', 'C': [1e10, 1e-5, 1e-5]})
    >>> np.abs(df.karmacode('0').instructions[0].rightfactor).sum() > 0.1
    True
    >>> np.abs(df.karmacode('0').instructions[2].rightfactor).sum() < 0.01
    True
    """
    if not isinstance(X, VirtualHStack):
        X = VirtualHStack(X, nb_threads=nb_threads)

    C = X.adjust_array_to_total_dimension(C, 'C')

    if w_warm is None:
        w_warm = np.zeros(X.shape[1] + 1, dtype=np.float)

    try:
        w0, obj_value, conv_dict = fmin_l_bfgs_b(logistic_loss_and_grad, w_warm, fprime=None,
                                                 args=(X, 2. * y.astype(bool) - 1, 1. / C, sample_weight),
                                                 iprint=0, pgtol=1e-7, maxiter=max_iter)
        conv_dict['objective_value'] = obj_value
        intercept, beta = w0[-1], w0[:-1]
        linear_pred = X.dot(beta)
        linear_pred += intercept
        betas = X.split_by_dims(beta)

        if KarmaSetup.verbose:
            print(conv_dict)

    except Exception as ee:
        X._close_pool()  # avoid memory leak
        raise ee

    X._close_pool()  # Warning should called manually at the exit from class
    return linear_pred, intercept, betas, conv_dict


@regression_on_blocks
def logistic_coefficients_fall_back(X, y, max_iter, solver='liblinear', C=1e10, sample_weight=None):
    # XXX : max_iter is not active for solver='liblinear'
    try:
        C = float(C)
    except TypeError:
        raise ValueError('parameter \'C\' is invalid: expected float for a {} solver, got {}'
                         .format(solver, C))

    lr = LogisticRegression(C=C, tol=1e-5, solver=solver, fit_intercept=True, max_iter=max_iter,
                            random_state=X.shape[0])
    lr.fit(X, y, sample_weight=sample_weight)
    return expit(lr.decision_function(X)), lr.intercept_[0], lr.coef_[0], {}


def logistic_coefficients(X, y, max_iter, solver='liblinear', C=1e10,
                          w_warm=None, sample_weight=None, nb_threads=1):
    if solver != 'lbfgs':
        return logistic_coefficients_fall_back(X, y, max_iter, solver,
                                               C=C, sample_weight=sample_weight)
    else:
        linear_pred, intercept, beta, conv_dict = logistic_coefficients_lbfgs(X, y, max_iter=max_iter, C=C,
                                                                              w_warm=w_warm,
                                                                              sample_weight=sample_weight,
                                                                              nb_threads=nb_threads)

        return expit(linear_pred), intercept, beta, conv_dict



def logistic_coefficients_and_posteriori(X, y, max_iter, w_priori=None, intercept_priori=0.,
                                         C_priori=1e10, intercept_C_priori=1e10,
                                         sample_weight=None, w_warm=None, nb_threads=1,
                                         full_hessian=True, timer=None):
    if timer is None:
        timer = create_timer(None)

    with timer('BayLogReg_Reg_Init'):
        if not isinstance(X, VirtualHStack):
            X = VirtualHStack(X, nb_threads=nb_threads)

        if w_priori is None:
            w_priori = np.zeros(X.shape[1], dtype=np.float)

        C_priori = X.adjust_array_to_total_dimension(C_priori, 'C_priori')
        w_priori = X.adjust_array_to_total_dimension(w_priori, 'w_priori')

        if w_warm is None:
            w_warm = np.hstack([w_priori, intercept_priori])

        alpha_priori = 1. / C_priori
        alpha_intercept = 1. / intercept_C_priori
        y = 2. * y.astype(bool) - 1
    try:
        with timer('BayLogReg_Reg_Mean'):
            w0, obj_value, conv_dict = fmin_l_bfgs_b(logistic_loss_and_grad, w_warm, fprime=None,
                                                     args=(X, y, alpha_priori, sample_weight, w_priori,
                                                           alpha_intercept, intercept_priori),
                                                     iprint=0, pgtol=1e-7, maxiter=max_iter)
            intercept, beta = w0[-1], w0[:-1]
            conv_dict['objective_value'] = obj_value

        with timer('BayLogReg_Reg_Variance'):
            if full_hessian:
                # full_hessian = hessian(w0, X, y, alpha_priori, sample_weight, alpha_intercept)
                C_post = diagonal_of_inverse_symposdef(hessian(w0, X, y, alpha_priori,
                                                               sample_weight, alpha_intercept))
            else:
                C_post = 1. / diag_hessian(w0, X, y, alpha_priori, sample_weight, alpha_intercept)

        intercept_C_post, feature_C_post = C_post[-1], C_post[:-1]

        linear_pred = X.dot(beta) + intercept
        betas = X.split_by_dims(beta)
        feature_C_posts = X.split_by_dims(feature_C_post.astype(np.float))
    except Exception as ee:
        X._close_pool()  # avoid memory leak
        raise ee

    X._close_pool()  # Warning should called manually at the exit from class
    return (expit(linear_pred),
            intercept, betas,
            intercept_C_post, feature_C_posts, conv_dict)


def hessian(w, X, y, alpha, sample_weight=None, alpha_intercept=0.):
    """
    >>> from sklearn.linear_model.logistic import _logistic_grad_hess
    >>> from karma.core.utils.utils import use_seed
    >>> with use_seed(123): X = VirtualHStack([np.random.rand(1000, 10).astype(np.float32)])
    >>> with use_seed(123): y = 2 * np.random.randint(0, 2, 1000) - 1.
    >>> with use_seed(123): w = np.random.rand(10 + 1)
    >>> alpha = 3.
    >>> HH_new = hessian(w, X, y, alpha)

    >>> _, Hlambda = _logistic_grad_hess(w, X.X[0], y, alpha)
    >>> HH_old = np.zeros((11, 11), dtype=np.float32)
    >>> s = np.eye(11, dtype=np.float32)
    >>> for i in xrange(11): HH_old[i] = Hlambda(s[i])
    >>> np.max(np.abs(HH_old - HH_new)) < 1e-5
    True

    """
    n_samples, n_features = X.shape

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    fit_intercept = (w.size == n_features + 1)
    c, w = (w[-1], w[:-1]) if fit_intercept else (0, w)

    yz = X.dot(w)
    yz += c
    z = expit(yz)

    weight = sample_weight * z * (1 - z)

    def sc(x):
        if is_karmasparse(x):
            x = x.tocsc()  # that does make copy
            x.scale_along_axis_inplace(weight, axis=1)
            return x
        else:
            return weight[:, np.newaxis] * x

    # Compute full Hessian
    size = n_features + 1 if fit_intercept else n_features
    Hess = np.zeros((size, size), dtype=np.float32)

    if fit_intercept:
        Hess[-1, -1] = weight.sum() + alpha_intercept
    SubHess = Hess[:n_features, :n_features]

    if X.is_block:
        def _hess_fill(i):
            dx = sc(X.X[i])
            if fit_intercept:
                Hess[X.dims[i]:X.dims[i + 1], -1] = dx.sum(axis=0)
            for j, xj in enumerate(X.X):
                if j >= i:
                    # TODO : downcast dtype for dense * dense dtype
                    Hess[X.dims[i]:X.dims[i + 1],
                         X.dims[j]:X.dims[j + 1]] = np.asarray(dx.transpose().dot(xj))

        with blas_threads(X.nb_inner_threads), open_mp_threads(X.nb_inner_threads):
            X.pool.map(_hess_fill, X.order)
        # symmetrize
        ind = np.triu_indices_from(Hess, 1)
        Hess[ind[::-1]] = Hess[ind]
    else:
        dx = sc(X.X)
        SubHess[:] = X.X.transpose().dot(dx)
        if fit_intercept:
            Hess[:-1, -1] = dx.sum(axis=0)
            Hess[-1, :-1] = Hess[:-1, -1]

    # set a priori
    np.fill_diagonal(SubHess, np.diag(SubHess) + alpha)
    return Hess


def diag_hessian(w, X, y, alpha, sample_weight=None, alpha_intercept=0.):
    """
    >>> from sklearn.linear_model.logistic import _logistic_grad_hess
    >>> from karma.core.utils.utils import use_seed
    >>> with use_seed(123): X = VirtualHStack([np.random.rand(1000, 10).astype(np.float32)])
    >>> with use_seed(123): y = 2 * np.random.randint(0, 2, 1000) - 1.
    >>> with use_seed(123): w = np.random.rand(10 + 1)
    >>> alpha = 3.
    >>> HH_diag = diag_hessian(w, X, y, alpha)

    >>> _, Hlambda = _logistic_grad_hess(w, X.X[0], y, alpha)
    >>> HH_old = np.zeros((11, 11), dtype=np.float32)
    >>> s = np.eye(11, dtype=np.float32)
    >>> for i in xrange(11): HH_old[i] = Hlambda(s[i])
    >>> np.max(np.abs(np.diag(HH_old) - HH_diag)) < 1e-5
    True

    """
    _, n_features = X.shape

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    fit_intercept = (w.size == n_features + 1)
    c, w = (w[-1], w[:-1]) if fit_intercept else (0, w)

    yz = X.dot(w)
    yz += c
    z = expit(yz)

    weight = sample_weight * z * (1 - z)

    def ssc(x):
        if is_karmasparse(x):
            return x.scale_along_axis(np.sqrt(weight), axis=1).sum_power(2, axis=0)
        else:
            return np.einsum('ji, ji, j->i', x, x, weight)

    size = n_features + 1 if fit_intercept else n_features
    DHess = np.zeros(size, dtype=np.float32)

    if fit_intercept:
        DHess[-1] = weight.sum() + alpha_intercept

    if X.is_block:
        def _dhess_fill(i):
            DHess[X.dims[i]:X.dims[i + 1]] = ssc(X.X[i])

        with blas_threads(X.nb_inner_threads), open_mp_threads(X.nb_inner_threads):
            X.pool.map(_dhess_fill, X.order)
    else:
        DHess[:n_features] = ssc(X.X)

    DHess[:n_features] += alpha
    return DHess
