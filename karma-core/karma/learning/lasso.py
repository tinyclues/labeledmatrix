#
# Copyright tinyclues, All rights reserved
#

import numpy as np
# XXX we should have used LassoLarsCV here but its parallel version is currently broken
from sklearn.linear_model import LassoCV

from karma.learning.lasso_gram import lasso_gram
from karma import KarmaSetup
from karma.thread_setter import blas_threads

N_fold_test = 3  # for auto model_rank detection

__all__ = ['best_model_cv', 'lassopath']


def compute_indices_and_model_rank(dataframe, x, y, model_rank, min_model_rank, cv=None, nb_threads=None):
    """
    >>> from karma.synthetic.regression import regression_dataframe
    >>> base_indices = np.arange(10)
    >>> ori_betas = np.arange(10.)

    # small fixed 'noise' on zero values that should be discarded by best-rank search,
    # but that discriminates between zero indices for deterministic testing
    >>> ori_betas[:3] = 1e-8 * np.arange(3)
    >>> dd = regression_dataframe(200, dim=10, coeff_dist=np.random.uniform, law=ori_betas,
    ...                           noise=0., logistic=False, missing_rate=0)

    # model_rank is the full set of variables
    >>> indices, model_rank = compute_indices_and_model_rank(dd, "x", "y", 20, None)
    >>> model_rank == 10
    True
    >>> np.all(indices == np.arange(10))
    True

    # fixed model_rank
    >>> indices, model_rank = compute_indices_and_model_rank(dd, "x", "y", 5, None)
    >>> model_rank == 5
    True
    >>> len(indices) == model_rank
    True

    # best model_rank
    >>> indices, model_rank = compute_indices_and_model_rank(dd, "x", "y", None, None )
    >>> model_rank == 7
    True
    >>> np.all(indices == np.arange(3,10))
    True

    # best model_rank with cv
    >>> from sklearn.cross_validation import KFold
    >>> cv = KFold(200, 5)
    >>> indices_, model_rank_ = compute_indices_and_model_rank(dd, "x", "y", None, None, cv=cv)
    >>> model_rank == model_rank_
    True
    >>> np.all(indices == indices_)
    True

    # min_model_rank under true rank
    >>> indices__, model_rank__ = compute_indices_and_model_rank(dd, "x", "y", None, 4)
    >>> model_rank == model_rank__
    True
    >>> np.all(indices == indices__)
    True

    # right now min_model_rank is not strictly enforced
    # min_model_rank over true rank
    >>> indices, model_rank = compute_indices_and_model_rank(dd, "x", "y", None, 8)
    >>> model_rank >= 7
    True

    # recompute with fixed rank gives same model
    >>> indices_guessed, model_rank_guessed = compute_indices_and_model_rank(dd, "x", "y", None, None, cv=cv)
    >>> np.all(compute_indices_and_model_rank(dd, "x", "y", model_rank_guessed, None, cv=cv)[0] == indices_guessed)
    True
    """
    dim = dataframe[x].vectorial_shape()[0]
    if model_rank is None:
        # no fixed model_rank input, find the best one
        indices, betas, intercepts = best_model_cv(dataframe, x, y, cv=cv, n_jobs=nb_threads)
        model_rank = indices.shape[0]
        if min_model_rank and model_rank < min_model_rank:
            # best model_rank is less than required minimum,
            # we need to compute the lasso path up until this minimum rank to have the correct indices
            # ideally this should be done in best_model_cv directly. Unfortunately the underlying sklearn class
            # model isn't nicely suited for this task. To investigate?
            betas, intercepts = lassopath(dataframe, x, y, min_model_rank)
            model_rank = max(min(model_rank, len(betas) - 1), 0)
            indices = betas[model_rank].nonzero()[0]
    elif model_rank < dim:
        betas, intercepts = lassopath(dataframe, x, y, model_rank)
        model_rank = max(min(model_rank, len(betas) - 1), 0)
        indices = betas[model_rank].nonzero()[0]
    else:  # full set of variables is used here!
        model_rank = dim
        indices = np.arange(dim, dtype=np.int)
    if KarmaSetup.verbose:
        print 'LOGISTIC_LASSO on column {} uses model_rank : {}'.format(x, model_rank)
        print "LOGISTIC_LASSO takes the following coordinates :"
        print [dataframe[x].coordinates[e] for e in indices]
    return indices, model_rank


def best_model_cv(dataframe, x, y, cv=None, n_jobs=None):
    """
    Compute the indices of the relevant variables using LARS-lasso (see Elements of Statistical Learning around p68).
    Non-zero indices

    >>> from karma.synthetic.regression import regression_dataframe
    >>> from karma import DataFrame, Column
    >>> base_indices = np.arange(10)
    >>> np.random.shuffle(base_indices)
    >>> zero_indices, non_zero_indices = np.sort(base_indices[:3]), np.sort(base_indices[3:])
    >>> ori_betas = np.random.uniform(1, size=10)
    >>> ori_betas[zero_indices] = 0.
    >>> dd = regression_dataframe(200, dim=10, coeff_dist=np.random.uniform, law=ori_betas,
    ...                           noise=0., logistic=False, missing_rate=0)
    >>> indices, betas, intercepts = best_model_cv(dd, "x", "y")
    >>> betas.shape == (10,)
    True
    >>> np.all(indices == non_zero_indices)
    True
    >>> err1 = np.std(dd['x'][:].dot(betas) + intercepts - dd['y'][:])
    >>> err2 = np.std(dd['y'][:])
    >>> err1 < 0.7 * err2
    True
    >>> from sklearn.cross_validation import KFold
    >>> cv = KFold(200, n_folds=4)
    >>> indices, betas, intercept = best_model_cv(dd, "x", "y", cv=cv)
    >>> betas.shape == (10,)
    True
    >>> np.all(indices == non_zero_indices)
    True

    # constant input
    >>> ddd = DataFrame(list(0.2 * np.ones((10,3,3))), columns=['a', 'b', 'c'])
    >>> ddd['y'] = Column(np.arange(10))
    >>> indices, betas, intercept = best_model_cv(ddd, "a", "y", cv=cv)
    >>> indices.shape, betas.shape
    ((0,), (0,))
    >>> intercept
    4.5

    # constant output
    >>> ddd = DataFrame(list(np.random.rand(10,3,3)), columns=['a', 'b', 'c'])
    >>> ddd['y'] = Column(np.ones(10, dtype='float'))
    >>> indices, betas, intercept = best_model_cv(ddd, "a", "y", cv=cv)
    >>> indices.shape, betas.shape
    ((0,), (0,))
    >>> intercept
    1.0
    """
    X = np.asarray(dataframe[x][:], dtype=np.float)
    Y = np.asarray(dataframe[y][:], dtype=np.float)
    if np.all(Y == Y[0]):
        return np.array([], dtype=np.int), np.array([], dtype=np.float), Y[0]
    if np.all(X == X[0, 0]):
        return np.array([], dtype=np.int), np.array([], dtype=np.float), Y.mean()
    with blas_threads(1 if cv is not None else None):
        if n_jobs is None:
            n_jobs = len(cv) if cv is not None else 1
        lasso_cv = LassoCV(cv=cv, normalize=True, precompute=True, n_jobs=n_jobs, random_state=len(X))
        lasso_cv.fit(X, Y)
    return np.nonzero(np.abs(lasso_cv.coef_) > 1e-10)[0], lasso_cv.coef_, lasso_cv.intercept_


def lassopath(dataframe, x, y, max_iter=None):
    """
    >>> from karma.synthetic.regression import regression_dataframe
    >>> dd = regression_dataframe(200, dim=10, coeff_dist=np.random.uniform, law=None,
    ...                           noise=0.10, logistic=False, missing_rate=0)
    >>> k = 7
    >>> betas, intercepts = lassopath(dd, "x", "y", k)
    >>> betas.shape == (k+1, 10)
    True
    >>> intercepts.shape == (k+1,)
    True
    >>> err1 = np.std(dd['x'][:].dot(betas[-1]) + intercepts[0] - dd['y'][:])
    >>> err2 = np.std(dd['y'][:])
    >>> err1 < 0.7 * err2
    True
    """

    def lassopath_from_moments(nblines, sum_x, sum_y, sum_xx, sum_xy, sum_yy):
        csum_xx, csum_xy, c_xx, norm_xy = _center_and_normalize(nblines, sum_x, sum_y,
                                                                sum_xx, sum_xy, sum_yy)
        cbetas = lasso_gram(csum_xy, csum_xx, max_iter=max_iter)[2]
        betas = _denormalized_beta(cbetas, c_xx, norm_xy)
        intercepts = _intercept(betas, sum_x, sum_y, nblines)
        return betas.T, intercepts

    def _center_and_normalize(nblines, sum_x, sum_y, sum_xx, sum_xy, sum_yy):
        #type conversion
        sum_x = np.atleast_2d(sum_x)
        sum_y = np.atleast_2d(sum_y)
        #computing centered variables
        c_xx = sum_xx - np.dot(sum_x.T, sum_x) / nblines
        c_xy = (sum_xy - np.dot(sum_y.T, sum_x) / nblines)[0]
        #normalizing
        m, n = c_xx.shape
        cn_xx = np.ones((m, n), dtype=np.float)
        norm_xy = np.sqrt(sum_yy - sum_y ** 2 / nblines)[0]
        # TODO: We should vectorize this loop
        cn_xy = c_xy / np.maximum(norm_xy, 1e-8)
        for i in xrange(0, m):
            cn_xy[i] /= np.maximum(np.sqrt(c_xx[i, i]), 1e-8)
            for j in xrange(i + 1, n):
                prod = c_xx[i, i] * c_xx[j, j]
                cn_xx[j, i] = cn_xx[i, j] = c_xx[i, j] / np.maximum(np.sqrt(prod), 1e-8)
        return cn_xx, cn_xy, c_xx, norm_xy

    def _denormalized_beta(cbetas, c_xx, norm_xy):
        cbetas *= norm_xy
        m, n = c_xx.shape
        for i in xrange(n):
            cbetas[i] /= np.maximum(np.sqrt(c_xx[i, i]), 1e-8)
        return cbetas

    def _intercept(beta, sum_x, sum_y, nblines):
        return (sum_y - np.dot(sum_x, beta)) / nblines
    # TODO : remove df.moments
    xx = np.asarray(dataframe[x][:], dtype=np.float)
    yy = np.asarray(dataframe[y][:], dtype=np.float)
    moments = [len(yy), xx.sum(axis=0), yy.sum(axis=0),
               xx.T.dot(xx), xx.T.dot(yy), yy.dot(yy)]
    max_iter = max_iter or 100
    return lassopath_from_moments(*moments)
