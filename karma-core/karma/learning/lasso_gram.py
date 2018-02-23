#
# Copyright tinyclues, All rights reserved
#

"""
Least Angle Regression algorithm based only on Covariance matrix
rebuilt from  scikit-learn/sklearn/linear_model/least_angle.py
"""

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from sklearn.utils import arrayfuncs


def lasso_gram(Xy, XX, max_features=None, max_iter=500, min_alpha=0, method='lasso'):

    """ Compute Least Angle Regression and LASSO path

Parameters
-----------
Xy: array, shape (n_features,),
  Precomputed Xy covariance matrix (X' * y)

XX: array, shape: (n_features,   n_features),
  Precomputed Gram matrix (X' * X)

max_features: integer, optional
Maximum number of selected features.

alpha_min: float, optional

Minimum correlation along the path. It corresponds to the
regularization parameter alpha parameter in the Lasso.

method: 'lar' | 'lasso'
Specifies the returned model. Select 'lar' for Least Angle
Regression, 'lasso' for the Lasso.

Returns
--------
alphas: array, shape: (max_features + 1,)
Maximum of covariances (in absolute value) at each
iteration.

active: array, shape (max_features,)
Indices of active variables at the end of the path.

coefs: array, shape (n_features, max_features+1)
Coefficients along the path

Exemple :
        # compare with sklearn version
        >>> from sklearn.linear_model import lars_path
        >>> from sklearn import datasets
        >>> diabetes = datasets.load_diabetes()
        >>> X, y = diabetes.data, diabetes.target
        >>> XX, Xy = np.dot(X.T, X), np.dot(X.T, y)
        >>> alphas, active, coef = lasso_gram(Xy, XX)
        >>> active
        [2, 8, 3, 1, 9, 4, 7, 5, 0, 6]
        >>> models  = lars_path(X, y, method = 'lasso') # algo 'lasso' from scikits
        >>> models[1] == active
        True
        >>> np.sum(np.abs(models[2] - coef)) < 0.00001
        True

        # random test
        >>> N, n = 100, 10
        >>> X = np.random.normal(size=(N, n))
        >>> y = X.dot(np.random.rand(n)) + 0.0001 * np.random.normal(size=N)
        >>> XX, Xy = np.dot(X.T, X), np.dot(X.T, y)
        >>> alphas, active, coef = lasso_gram(Xy, XX)
        >>> models  = lars_path(X, y, method = 'lasso') # algo 'lasso' from scikits
        >>> models[1] == active
        True
        >>> np.sum(np.abs(models[2] - coef)) < 0.00001
        True
"""
    Gram = np.array(XX, dtype=np.float)
    Cov = np.array(Xy, dtype=np.float)
    n_features = Gram.shape[0]
    if max_features is None:
        max_features = n_features

    coefs = np.zeros((max_features + 1, n_features))
    alphas = np.zeros(max_features + 1)
    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False
    eps = np.finfo(Gram.dtype).eps

    # will hold the cholesky factorization. Only lower part is referenced.
    L = np.empty((max_features, max_features), dtype=Gram.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (Gram,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (Gram,))

    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning

    while True:
        if Cov.size:
            C_idx = np.argmax(np.abs(Cov))
            C_ = Cov[C_idx]
            C = np.fabs(C_)
        else:
            C = 0.

        alpha = alphas[n_iter, np.newaxis]
        coef = coefs[n_iter]
        prev_alpha = alphas[n_iter - 1, np.newaxis]
        prev_coef = coefs[n_iter - 1]
        alpha[0] = C
        if alpha[0] <= min_alpha:  # early stopping
            if not alpha[0] == min_alpha:
                # interpolation factor 0 <= ss < 1
                if n_iter > 0:
                    # In the first iteration, all alphas are zero, the formula
                    # below would make ss a NaN
                    ss = ((prev_alpha[0] - min_alpha) /
                          (prev_alpha[0] - alpha[0]))
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = min_alpha
            coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= max_features:
            break

        if not drop:
            sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov_not_shortened = Cov
            Cov = Cov[1:]  # remove Cov[0]

            # swap does only work inplace if matrix is fortran
            # contiguous ...
            Gram[m], Gram[n] = swap(Gram[m], Gram[n])
            Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
            c = Gram[n_active, n_active]
            L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active],
                                        trans=0, lower=True,
                                        overwrite_b=True, check_finite=False)
            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            if diag < 1e-7:
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue

            active.append(indices[n_active])
            n_active += 1

        if method == 'lasso' and n_iter > 0 and prev_alpha[0] < alpha[0]:
            break

        # least squares solution
        least_squares, info = solve_cholesky(L[:n_active, :n_active],
                                             sign_active[:n_active],
                                             lower=True)

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, info = solve_cholesky(L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        # if huge number of features, this takes 50% of time, I
        # think could be avoided if we just update it using an
        # orthogonal (QR) decomposition of X
        corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                             least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
        gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            idx = np.where(z == z_pos)[0][::-1]
            sign_active[idx] = - sign_active[idx]

            if method == 'lasso':
                gamma_ = z_pos
            drop = True

        n_iter += 1

        if n_iter >= coefs.shape[0]:
            del coef, alpha, prev_alpha, prev_coef
            # resize the coefs and alphas array
            add_features = 2 * max(1, (max_features - n_active))
            coefs.resize((n_iter + add_features, n_features))
            alphas.resize(n_iter + add_features)
        coef = coefs[n_iter]
        prev_coef = coefs[n_iter - 1]

        coef[active] = prev_coef[active] + gamma_ * least_squares

        # update correlations
        Cov -= gamma_ * corr_eq_dir

        # See if any coefficient has changed sign
        if drop and method == 'lasso':
            [arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii) for ii in idx]
            n_active -= 1
            m, n = idx, n_active
            drop_idx = [active.pop(ii) for ii in idx]
            for ii in idx:
                for i in range(ii, n_active):
                    indices[i], indices[i + 1] = indices[i + 1], indices[i]
                    Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i+1])
                    Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i], Gram[:, i + 1])

            temp = Xy[drop_idx] - np.dot(XX[drop_idx], coefs[n_iter])
            Cov = np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size

    alphas = alphas[:n_iter + 1]
    coefs = coefs[:n_iter + 1]
    return alphas, active, coefs.T
