#
# Copyright tinyclues, All rights reserved
#
from functools import wraps

import numpy as np
from cyperf.matrix.karma_sparse import is_karmasparse
from sklearn.linear_model import Ridge

from karma.core.curve import compute_mean_curve, gain_from_prediction, nonbinary_gain_from_prediction
from karma.core.utils.array import is_binary
from karma.learning.matrix_utils import safe_hstack


def regression_on_blocks(regression_method):
    @wraps(regression_method)
    def decorated_function(X, y, *args, **kwargs):
        is_block = isinstance(X, (list, tuple))
        if is_block:
            dims = np.cumsum([0] + [x.shape[1] for x in X])
            # FIXME Ridge and LogisticRegression classes doesn't support different regularizations
            # for different features so if we want to do such selective regularization a workaround need to be found
            # C = kwargs.get('C', 1e10)
            # if isinstance(C, (list, tuple, np.ndarray)):
            #     if len(C) == len(X):
            #         kwargs['C'] = np.concatenate([np.full(x.shape[1], C[i]) for i, x in enumerate(X)])
        XX = (X[0] if len(X) == 1 else safe_hstack(X)) if is_block else X

        result = regression_method(XX, y, *args, **kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            y_hat, intercept, beta = result
            if is_block:
                beta = [beta[dims[i]:dims[i + 1]] for i in xrange(len(dims) - 1)]
            return y_hat, intercept, beta
        else:
            return result

    return decorated_function


@regression_on_blocks
def linear_regression_coefficients(X, y, intercept=True, C=1e10):
    lr = Ridge(fit_intercept=intercept, alpha=1 / np.asarray(C, dtype=np.float), tol=0.0001)
    lr.fit(X, y)

    return lr.predict(X), lr.intercept_, lr.coef_.T


@regression_on_blocks
def sklearn_regression_model(X, y, regression_class, **kwargs):
    if 'max_features' in kwargs:
        kwargs['max_features'] = min(X.shape[1], kwargs['max_features'])

    if is_karmasparse(X):
        X = X.to_scipy_sparse(copy=False)

    clf = regression_class(random_state=X.shape[0], **kwargs)
    clf.fit(X, y)

    return clf.predict(X), clf


def create_meta_of_regression(prediction, y, test_curves=None, train_mse=None, test_mses=None):
    binary_prediction = is_binary(y) and np.max(prediction) <= 1. and np.min(prediction) >= 0.
    if binary_prediction:
        curve_method = gain_from_prediction
    elif np.min(y) >= 0:
        curve_method = nonbinary_gain_from_prediction
    else:
        curve_method = None

    meta = {}
    if curve_method is not None:
        curves = curve_method(prediction, y)
        if binary_prediction:
            curves.guess = gain_from_prediction(prediction)

        if test_curves is not None:
            curves.tests = test_curves
            curves.test = compute_mean_curve(test_curves)

        meta['curves'] = curves
    if train_mse is not None:
        meta['train_MSE'] = train_mse
    if test_mses is not None:
        meta['test_MSEs'] = test_mses

    return meta
