from collections import OrderedDict
from copy import deepcopy
from time import time

import numexpr as ne
import numpy as np
from cyperf.matrix.karma_sparse import is_karmasparse
from cyperf.tools.getter import build_safe_decorator
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import truncnorm
from sklearn.linear_model.logistic import LogisticRegression

from karma.core.utils import create_timer, use_seed
from karma.learning.matrix_utils import diagonal_of_inverse_symposdef
from karma.learning.regression import regression_on_blocks
from karma.learning.utils import NB_THREADS_MAX
from karma.learning.utils import VirtualHStack, VirtualDirectProduct, vdp_materilaze
from karma.runtime import KarmaSetup
from karma.thread_setter import blas_threads, open_mp_threads

__all__ = ['logistic_coefficients', 'logistic_coefficients_and_posteriori', 'expit',
           'CONVERGENCE_INFO_STATUS', 'CONVERGENCE_INFO_DESIGN_WIDTH']

CONVERGENCE_INFO_STATUS = 'status'
CONVERGENCE_INFO_DESIGN_WIDTH = 'width'


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

    yz = X.dot(w)
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
                          w_warm=None, sample_weight=None, nb_threads=1,
                          nb_inner_threads=None, verbose=True):
    if solver != 'lbfgs':
        return logistic_coefficients_fall_back(X, y, max_iter, solver, C=C, sample_weight=sample_weight)
    else:
        logist_pred, intercept, beta, _, _, conv_dict = logistic_coefficients_and_posteriori(
            X, y,
            max_iter=max_iter,
            w_priori=None,
            intercept_priori=0,
            C_priori=C,
            intercept_C_priori=1e10,
            sample_weight=sample_weight,
            w_warm=w_warm,
            nb_threads=nb_threads,
            nb_inner_threads=nb_inner_threads,
            hessian_mode='skip',
            verbose=verbose,
        )

        return logist_pred, intercept, beta, conv_dict


def logistic_coefficients_and_posteriori(X, y,
                                         max_iter,
                                         w_priori=None, intercept_priori=0.,
                                         C_priori=1e10, intercept_C_priori=1e10,
                                         sample_weight=None,
                                         w_warm=None,
                                         nb_threads=1,
                                         nb_inner_threads=None,
                                         hessian_mode='full',
                                         timer=None,
                                         verbose=True):
    """
    >>> from karma.core.dataframe import DataFrame
    >>> from karma.core.utils.utils import use_seed
    >>> with use_seed(123):
    ...     df = DataFrame({'a': np.random.poisson(15, size=(1000, 10)), 'b': np.random.poisson(15, size=(1000, 3)),
    ...                     'c': np.random.poisson(15, size=(1000, 14)), 'y': np.random.randint(0, 2, 1000)})
    >>> with use_seed(1237):
    ...     res1 = df.build_lib_column('logistic_regression', ('a', 'b', 'c'),
    ...                 {'solver': 'lbfgs', 'axis': 'y', 'verbose': False,
    ...                  'C': np.array([10.] * 10 + [1000.] * 3 + [10000000.] * 14)})
    >>> with use_seed(1237):
    ...     res2 = df.build_lib_column('logistic_regression', ('a', 'b', 'c'),
    ...                 {'solver': 'lbfgs', 'axis': 'y', 'verbose': False,
    ...                  'C': [10., 1000., 10000000.]})
    >>> np.array_equal(res1[:], res2[:])
    True
    >>> df['0'] = df.build_lib_column('logistic_regression', ('a', 'b', 'c'),
    ...                 {'solver': 'lbfgs', 'axis': 'y', 'verbose': False, 'C': [1e10, 1e-5, 1e-5]})
    >>> np.abs(df.karmacode('0').instructions[0].rightfactor).sum() > 0.1
    True
    >>> np.abs(df.karmacode('0').instructions[2].rightfactor).sum() < 0.01
    True
    """

    if timer is None:
        timer = create_timer(None)

    with timer('LogReg_Reg_Init'):
        if not isinstance(X, VirtualHStack):
            X = VirtualHStack(X, nb_threads=nb_threads, nb_inner_threads=nb_inner_threads)

        C_priori = X.adjust_array_to_total_dimension(C_priori, 'C_priori')

        if w_priori is None:
            w_priori = np.zeros(X.shape[1], dtype=np.float)
        else:
            w_priori = X.adjust_array_to_total_dimension(w_priori, 'w_priori')

        if w_warm is None:
            w_warm = np.hstack([w_priori, intercept_priori])
            with use_seed(seed=X.shape[1]):
                # random perturbation of starting point
                w_warm[:-1] += np.minimum(C_priori, 1) * truncnorm(-1, 1).rvs(w_priori.shape[0])

        alpha_priori = 1. / C_priori
        alpha_intercept = 1. / intercept_C_priori
        y = 2. * y.astype(bool) - 1
    try:
        with timer('LogReg_Reg_Mean'):
            lbfgs_params = KarmaSetup.lbfgs_params
            start_lbfgs = time()
            w0, obj_value, conv_dict = fmin_l_bfgs_b(logistic_loss_and_grad, w_warm, fprime=None,
                                                     args=(X, y, alpha_priori, sample_weight,
                                                           w_priori, alpha_intercept, intercept_priori),
                                                     iprint=0, pgtol=lbfgs_params.get('pgtol', 1e-7),
                                                     maxiter=max_iter, m=lbfgs_params.get('m', 100),
                                                     factr=lbfgs_params.get('factr', 1e7),
                                                     maxfun=lbfgs_params.get('maxfun', 15000),
                                                     maxls=lbfgs_params.get('maxls', 20))
            end_lbfgs = time()
            lbfgs_timing = round(end_lbfgs - start_lbfgs, 2)

        conv_dict = _conv_dict_format(conv_dict, obj_value, X.shape[0], nb_threads, nb_inner_threads, lbfgs_timing,
                                      verbose)

        intercept, beta = w0[-1], w0[:-1]
        linear_pred = X.dot(beta) + intercept
        betas = X.split_by_dims(beta)

        with timer('LogReg_Reg_Variance'):
            if hessian_mode == 'skip':
                C_post = np.full(X.shape[1] + 1, np.nan, dtype=np.float)
            elif hessian_mode == 'full':
                C_post = diagonal_of_inverse_symposdef(hessian(w0, X, y, alpha_priori,
                                                               sample_weight, alpha_intercept))
            elif hessian_mode == 'diag':
                C_post = 1. / diag_hessian(w0, X, y, alpha_priori, sample_weight, alpha_intercept)
            else:
                raise ValueError('hessian_mode needs to be one of {{skip, full, diag}}, got {}'.format(hessian_mode))

        intercept_C_post, feature_C_post = C_post[-1], C_post[:-1]
        feature_C_posts = X.split_by_dims(feature_C_post.astype(np.float))
    finally:
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
        x = vdp_materilaze(x)
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

    with blas_threads(X.nb_inner_threads), open_mp_threads(X.nb_inner_threads):
        if X.is_block:
            def _hess_fill(i):
                dx = sc(X.X[i])
                if fit_intercept:
                    Hess[X.dims[i]:X.dims[i + 1], -1] = dx.sum(axis=0)
                for j, xj in enumerate(X.X):
                    if j >= i:
                        # TODO : downcast dtype for dense * dense dtype
                        Hess[X.dims[i]:X.dims[i + 1],
                        X.dims[j]:X.dims[j + 1]] = np.asarray(dx.T.dot(vdp_materilaze(xj)))

            if X.pool is not None:
                X.pool.map(_hess_fill, X.order)
            else:
                map(_hess_fill, X.order)

            ind = np.triu_indices_from(Hess, 1)
            Hess[ind[::-1]] = Hess[ind]
        else:
            mat_xx = vdp_materilaze(X.X)
            dx = sc(mat_xx)
            SubHess[:] = dx.T.dot(mat_xx)
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
        elif isinstance(x, VirtualDirectProduct):
            return x.transpose_dot(weight, power=2)
        else:
            return np.einsum('ji, ji, j->i', x, x, weight)

    size = n_features + 1 if fit_intercept else n_features
    DHess = np.zeros(size, dtype=np.float32)

    if fit_intercept:
        DHess[-1] = weight.sum() + alpha_intercept

    with blas_threads(X.nb_inner_threads), open_mp_threads(X.nb_inner_threads):
        if X.is_block:
            def _dhess_fill(i):
                DHess[X.dims[i]:X.dims[i + 1]] = ssc(X.X[i])

            if X.pool is not None:
                X.pool.map(_dhess_fill, X.order)
            else:
                map(_dhess_fill, X.order)
        else:
            DHess[:n_features] = ssc(X.X)

    DHess[:n_features] += alpha
    return DHess


@build_safe_decorator({})
def _conv_dict_format(conv_dict, obj_value, n_obs_design, nb_threads, nb_inner_threads, lbfgs_timing, verbose=True):
    """Pretty printing of lbfgs convergence information.
    """
    conv_dict_copy = deepcopy(conv_dict)
    conv_dict_copy['objective_value'] = obj_value

    gradient = conv_dict_copy.pop('grad', None)
    if gradient is None:
        norm_grad = None
        max_grad = None
    else:
        norm_grad = np.linalg.norm(gradient)
        max_grad = np.max(np.abs(gradient))
    conv_dict_copy['grad_l2_momentum'] = norm_grad / len(gradient)
    conv_dict_copy['grad_max'] = max_grad

    conv_dict_copy['stopping_criterion'] = conv_dict_copy.pop('task')

    warn_flag_translation_dict = {0: 'Convergence',
                                  1: 'No Convergence: too many function evaluations',
                                  2: 'No Convergence: stopping criterion not reached'}

    conv_dict_copy[CONVERGENCE_INFO_STATUS] = warn_flag_translation_dict[conv_dict['warnflag']]
    conv_dict_copy[CONVERGENCE_INFO_DESIGN_WIDTH] = len(gradient)

    conv_dict_copy['nit'] = conv_dict['nit']
    conv_dict_copy['n_funcalls'] = conv_dict['funcalls']
    conv_dict_copy['height'] = n_obs_design
    conv_dict_copy['outer_thr'] = nb_threads or 1

    if nb_inner_threads is None:
        if nb_threads is None or nb_threads == 1:
            conv_dict_copy['inner_thr'] = NB_THREADS_MAX
        else:
            conv_dict_copy['inner_thr'] = min(NB_THREADS_MAX, max(1, int(2 * 32. / nb_threads)))
    else:
        conv_dict_copy['inner_thr'] = nb_inner_threads

    conv_dict_copy['time_by_iteration'] = lbfgs_timing / float(conv_dict['nit'])
    conv_dict_copy['cols_to_rows_ratio'] = conv_dict_copy['width'] / float(n_obs_design)

    ordered_keys = [CONVERGENCE_INFO_STATUS, 'nit', 'grad_l2_momentum', 'grad_max',
                    CONVERGENCE_INFO_DESIGN_WIDTH, 'height', 'outer_thr', 'inner_thr', 'time_by_iteration']
    conv_dict_ordered = OrderedDict([(key, conv_dict_copy[key]) for key in ordered_keys])

    if KarmaSetup.verbose or verbose:
        from karma.core.dataframe import DataFrame
        DataFrame(conv_dict_ordered, one_line=True).preview()

    return conv_dict_ordered
