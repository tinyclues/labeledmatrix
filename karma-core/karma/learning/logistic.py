from collections import OrderedDict
from copy import deepcopy
from functools import wraps
from time import time

from six.moves import range

import numexpr as ne
import numpy as np
from numpy.linalg import norm
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


def _logistic_loss_and_grad(w, intercept, X, y, alpha, w_priori, alpha_intercept, intercept_priori, sample_weight):
    """
    Classic LogLoss penalty with L2 penalty

    :param w: betas to evaluate the function at, shape (n_features,)
    :param intercept: intercept to evaluate the function at, scalar
    :param X: VirtualHStack of shape (n_samples, n_features)
    :param y: binary np.ndarray of length n_samples (taking values in {-1, 1})
    :param alpha: L2 penalty for the betas, shape (n_features,)
    :param w_priori: prior for the betas, shape (n_features,)
    :param alpha_intercept: L2 penalty for the intercept, scalar
    :param intercept_priori: prior for the intercept, scalar
    :param sample_weight: weight vector for the sample, shape (n_sample,); assumes equal weight if None

    :return: loss, grad
        loss -> scalar
        grad -> same shape as w (n_features,)
        grad_intercept -> scalar
    """
    n_samples, n_features = X.shape

    yz = X.dot(w)
    yz += intercept
    yz *= y
    z0 = _expit_m1_times_y(yz, y)

    dw = alpha * (w - w_priori)
    loss = .5 * dw.dot(w - w_priori)

    if sample_weight is None:
        loss -= _log_logistic_inplace(yz).sum()
    else:
        loss -= _log_logistic_inplace_sample_weight(yz, sample_weight).sum()
        z0 *= sample_weight

    grad = X.transpose_dot(z0)
    grad += dw

    grad_intercept = z0.sum() + alpha_intercept * (intercept - intercept_priori)
    return loss, grad, grad_intercept


def wrap_with_logger_and_callback(func, log=False):
    if log:
        convergence_logs, _logs = [], {'loss': 0, 'grad': []}

        @wraps(func)
        def decorated_function(*args, **kwargs):
            loss, grad = func(*args, **kwargs)
            _logs['loss'] = loss
            _logs['grad'] = grad
            return loss, grad

        def callback(betas):
            infos = {
                'loss': _logs['loss'],
                'grad_l2_momentum': norm(_logs['grad'], 2) / len(_logs['grad']),
                'grad_max': norm(_logs['grad'], np.inf),
                'step_sum': None,
                'step_l1': None,
                'step_max': None,
            }
            if 'prev_betas' in _logs:
                step = betas - _logs['prev_betas']
                infos.update({
                    'step_sum': np.sum(step),
                    'step_l1': norm(step, 1),
                    'step_max': norm(step, np.inf)
                })
            _logs['prev_betas'] = betas
            convergence_logs.append(infos)

        return decorated_function, callback, convergence_logs
    else:
        return func, None, None


def logistic_loss_and_grad(w, X, y, alpha, w_priori, alpha_intercept, intercept_priori, sample_weight=None):
    """
    Classic LogLoss penalty with L2 penalty

    :param w: betas and intercept to evaluate the function at, shape (n_features + 1,)
    :param X: VirtualHStack of shape (n_samples, n_features)
    :param y: binary np.ndarray of length n_samples (taking values in {-1, 1})
    :param alpha: L2 penalty for the betas, shape (n_features,)
    :param w_priori: prior for the betas, shape (n_features,)
    :param alpha_intercept: L2 penalty for the intercept, scalar
    :param intercept_priori: prior for the intercept, scalar
    :param sample_weight: weight vector for the sample, shape (n_sample,); assumes equal weight if None

    :return: loss, grad
        loss -> scalar
        grad -> same shape as w (n_features + 1,)
    """
    loss, grad, grad_intercept = _logistic_loss_and_grad(w[:-1], w[-1], X, y,
                                                         alpha, w_priori,
                                                         alpha_intercept, intercept_priori,
                                                         sample_weight)
    return loss, np.hstack((grad, grad_intercept))


def logistic_loss_and_grad_elastic_net(w_extended, X, y,
                                       alpha, w_priori, alpha_intercept, intercept_priori,
                                       sample_weight=None, l1_coeff=0.2):
    """
    This adds an extra L1 penalty to the logistic_loss, as `l1_coef * |\alpha * (beta - beta_0)|_1`
        no L1 penalty is added on the intercept

    It works by splitting the betas into their positive and negative parts
        https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/andrew07scalable.pdf
        https://www.cs.ubc.ca/~murphyk/Papers/aistats09.pdf
        https://arxiv.org/pdf/1206.1156.pdf

    :param w_extended: betas and intercept to evaluate the function at, shape (2 * n_features + 1,):
        [positive_part] + [negative_part] + [intercept]
    :param X: VirtualHStack of shape (n_samples, n_features)
    :param y: binary np.ndarray of length n_samples (taking values in {-1, 1})
    :param alpha: L2 penalty for the betas, shape (n_features,)
    :param w_priori: prior for the betas (both for L2 and L1 penalties), shape (n_features,)
    :param alpha_intercept: L2 penalty for the intercept, scalar
    :param intercept_priori: prior for the intercept (for L2 penalty only), scalar
    :param sample_weight: weight vector for the sample, shape (n_sample,); assumes equal weight if None
    :param l1_coeff: extra L1 penalty

    :return: loss, grad
        loss -> scalar
        grad -> same shape as w_extended (2 * n_features + 1,)
    """
    n_samples, n_features = X.shape

    positive_slice, negative_slice = slice(0, n_features), slice(n_features, 2 * n_features)

    loss, signed_grad, grad_intercept = _logistic_loss_and_grad(
        w_priori + w_extended[positive_slice] - w_extended[negative_slice],
        w_extended[-1],
        X, y,
        alpha, w_priori,
        alpha_intercept, intercept_priori,
        sample_weight)

    loss += l1_coeff * alpha.dot(w_extended[positive_slice] + w_extended[negative_slice])

    grad = np.empty_like(w_extended)
    grad[positive_slice] = signed_grad
    grad[positive_slice] += l1_coeff * alpha
    grad[negative_slice] = - signed_grad
    grad[negative_slice] += l1_coeff * alpha

    grad[-1] = grad_intercept
    return loss, grad


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
        logist_pred, intercept, beta, _, _, conv_dict, _ = logistic_coefficients_and_posteriori(
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
                                         l1_coeff=0.,
                                         sample_weight=None,
                                         w_warm=None,
                                         nb_threads=1,
                                         nb_inner_threads=None,
                                         hessian_mode='full',
                                         timer=None,
                                         log=False,
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

        n_samples, n_features = X.shape
        row_nnz = X.row_nnz + 1  # to take into account the intercept
        C_priori = X.adjust_array_to_total_dimension(C_priori, 'C_priori')

        if w_priori is None:
            w_priori = np.zeros(n_features, dtype=np.float)
        else:
            w_priori = X.adjust_array_to_total_dimension(w_priori, 'w_priori')

        if w_warm is None:
            w_warm = np.hstack([w_priori, intercept_priori])
            with use_seed(seed=n_features):
                # random perturbation of starting point
                w_warm[:-1] += np.minimum(C_priori, 1) * truncnorm(-1, 1).rvs(n_features)

        alpha = 1. / C_priori
        alpha_intercept = 1. / intercept_C_priori
        y = 2. * y.astype(bool) - 1
    try:
        with timer('LogReg_Reg_Mean'):
            start_lbfgs = time()

            lbfgs_params = KarmaSetup.lbfgs_params
            l_bfgs_b_kwargs = dict(fprime=None,
                                   iprint=0,
                                   pgtol=lbfgs_params.get('pgtol', 1e-7),
                                   m=lbfgs_params.get('m', 100),
                                   factr=lbfgs_params.get('factr', 1e7),
                                   maxfun=lbfgs_params.get('maxfun', 15000),
                                   maxls=lbfgs_params.get('maxls', 20),
                                   maxiter=max_iter)

            if l1_coeff == 0:
                _loss, _callback, convergence_logs = wrap_with_logger_and_callback(logistic_loss_and_grad, log)
                w0, obj_value, conv_dict = fmin_l_bfgs_b(_loss, w_warm,
                                                         args=(X, y,
                                                               alpha, w_priori,
                                                               alpha_intercept, intercept_priori,
                                                               sample_weight),
                                                         callback=_callback,
                                                         **l_bfgs_b_kwargs)
                intercept, beta = w0[-1], w0[:-1]
            elif l1_coeff > 0:
                _loss, _callback, convergence_logs = wrap_with_logger_and_callback(logistic_loss_and_grad_elastic_net,
                                                                                   log)
                positive_slice, negative_slice = slice(0, n_features), slice(n_features, 2 * n_features)
                _w_warm_extended = np.zeros(2 * n_features + 1, dtype=np.float)
                _w_warm_extended[positive_slice] = np.maximum(w_warm[:-1] - w_priori, 0)
                _w_warm_extended[negative_slice] = np.maximum(w_priori - w_warm[:-1], 0)
                _w_warm_extended[-1] = w_warm[-1]
                bounds = [(0, None)] * (2 * n_features) + [(None, None)]
                _w0, obj_value, conv_dict = fmin_l_bfgs_b(_loss, _w_warm_extended,
                                                          args=(X, y,
                                                                alpha, w_priori,
                                                                alpha_intercept, intercept_priori,
                                                                sample_weight, l1_coeff),
                                                          bounds=bounds,
                                                          callback=_callback,
                                                          **l_bfgs_b_kwargs)
                intercept = _w0[-1]
                beta = w_priori + _w0[positive_slice] - _w0[negative_slice]
            else:
                raise AssertionError
            end_lbfgs = time()
            lbfgs_timing = np.round(end_lbfgs - start_lbfgs, 2)

        conv_dict = _conv_dict_format(conv_dict, obj_value, n_samples, nb_threads, nb_inner_threads, lbfgs_timing,
                                      row_nnz, verbose, convergence_logs, double_design_width=l1_coeff > 0)

        linear_pred = X.dot(beta) + intercept
        betas = X.split_by_dims(beta)

        with timer('LogReg_Reg_Variance'):
            if hessian_mode == 'skip':
                C_post = np.full(n_features + 1, np.nan, dtype=np.float)
            elif hessian_mode == 'full':
                C_post = diagonal_of_inverse_symposdef(hessian(np.hstack((beta, intercept)), X, y, alpha,
                                                               sample_weight, alpha_intercept))
            elif hessian_mode == 'diag':
                C_post = 1. / diag_hessian(np.hstack((beta, intercept)), X, y, alpha, sample_weight,
                                           alpha_intercept)
            else:
                raise ValueError('hessian_mode needs to be one of {{skip, full, diag}}, got {}'.format(hessian_mode))

        intercept_C_post, feature_C_post = C_post[-1], C_post[:-1]
        feature_C_posts = X.split_by_dims(feature_C_post.astype(np.float))
    finally:
        X._close_pool()  # Warning should called manually at the exit from class

    return (expit(linear_pred),
            intercept, betas,
            intercept_C_post, feature_C_posts, conv_dict, convergence_logs)


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
    >>> for i in range(11): HH_old[i] = Hlambda(s[i])
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
    >>> for i in range(11): HH_old[i] = Hlambda(s[i])
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
def _conv_dict_format(conv_dict, obj_value, n_obs_design, nb_threads, nb_inner_threads, lbfgs_timing, row_nnz,
                      verbose=True, logs=None, double_design_width=False):
    """Pretty printing of lbfgs convergence information.
    """
    conv_dict_copy = deepcopy(conv_dict)
    conv_dict_copy['objective_value'] = obj_value

    gradient = conv_dict_copy.pop('grad', None)
    if gradient is None:
        norm_grad = None
        max_grad = None
    else:
        norm_grad = norm(gradient, 2)
        max_grad = norm(gradient, np.inf)
    conv_dict_copy['grad_l2_momentum'] = norm_grad / len(gradient)
    conv_dict_copy['grad_max'] = max_grad

    conv_dict_copy['stopping_criterion'] = conv_dict_copy.pop('task')

    warn_flag_translation_dict = {0: 'Convergence',
                                  1: conv_dict_copy['stopping_criterion'].capitalize(),
                                  2: conv_dict_copy['stopping_criterion'].capitalize()}

    conv_dict_copy[CONVERGENCE_INFO_STATUS] = warn_flag_translation_dict[conv_dict['warnflag']]
    if double_design_width:
        # intercept is not l1 penalized and thus not doubled
        conv_dict_copy[CONVERGENCE_INFO_DESIGN_WIDTH] = (len(gradient) - 1) / 2 + 1
    else:
        conv_dict_copy[CONVERGENCE_INFO_DESIGN_WIDTH] = len(gradient)

    conv_dict_copy['nit'] = conv_dict['nit']
    conv_dict_copy['n_funcalls'] = conv_dict['funcalls']
    conv_dict_copy['height'] = n_obs_design
    conv_dict_copy['row_nnz'] = row_nnz
    conv_dict_copy['outer_thr'] = nb_threads or 1
    conv_dict_copy['mean_objective'] = obj_value / float(n_obs_design)

    if nb_inner_threads is None:
        if nb_threads is None or nb_threads == 1:
            conv_dict_copy['inner_thr'] = NB_THREADS_MAX
        else:
            conv_dict_copy['inner_thr'] = min(NB_THREADS_MAX, max(1, int(2 * 32. / nb_threads)))
    else:
        conv_dict_copy['inner_thr'] = nb_inner_threads

    conv_dict_copy['time_by_iteration'] = lbfgs_timing / float(conv_dict['nit'])
    conv_dict_copy['cols_to_rows_ratio'] = conv_dict_copy[CONVERGENCE_INFO_DESIGN_WIDTH] / float(n_obs_design)

    ordered_keys = [CONVERGENCE_INFO_STATUS, 'nit', 'grad_l2_momentum', 'grad_max', 'mean_objective', 'row_nnz',
                    CONVERGENCE_INFO_DESIGN_WIDTH, 'height', 'outer_thr', 'inner_thr', 'time_by_iteration']
    conv_dict_ordered = OrderedDict([(key, conv_dict_copy[key]) for key in ordered_keys])

    if KarmaSetup.verbose or verbose:
        from karma.core.dataframe import DataFrame
        DataFrame(conv_dict_ordered, one_line=True).preview()

    return conv_dict_ordered
