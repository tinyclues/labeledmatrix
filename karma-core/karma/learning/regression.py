#
# Copyright tinyclues, All rights reserved
#
from functools import wraps

import numpy as np
from sklearn.linear_model import Ridge

from karma.core.utils.utils import coerce_to_tuple_and_check_all_strings
from karma.core.curve import compute_mean_curve, gain_curve_from_prediction
from karma.core.utils.array import is_binary

__all__ = ['regression_on_blocks', 'linear_regression_coefficients',
           'create_meta_of_regression', 'create_summary_of_regression']


def regression_on_blocks(regression_method):
    @wraps(regression_method)
    def decorated_function(X, y, *args, **kwargs):
        from karma.learning.utils import BasicVirtualHStack
        if not isinstance(X, BasicVirtualHStack):
            X = BasicVirtualHStack(X)

        result = regression_method(X.materialize(), y, *args, **kwargs)

        if isinstance(result, tuple) and len(result) == 3:
            y_hat, intercept, beta = result
            return y_hat, intercept, X.split_by_dims(beta)
        else:
            return result

    return decorated_function


@regression_on_blocks
def linear_regression_coefficients(X, y, intercept=True, C=1e10, sample_weight=None):
    lr = Ridge(fit_intercept=intercept, alpha=1 / np.asarray(C, dtype=np.float), tol=0.0001)
    lr.fit(X, y, sample_weight)

    return lr.predict(X), lr.intercept_, lr.coef_.T


def create_meta_of_regression(prediction, y, with_guess=True, test_curves=None, test_mses=None, name=None):
    meta = {'train_MSE': np.mean((y - prediction) ** 2)}
    if np.min(y) >= 0:
        curves = gain_curve_from_prediction(prediction, y, name)
        if with_guess and is_binary(y) and np.max(prediction) <= 1 and np.min(prediction) >= 0:
            curves.guess = gain_curve_from_prediction(prediction)

        if test_curves is not None:
            if len(test_curves) > 1:
                curves.tests = test_curves
            curves.test = compute_mean_curve(test_curves)

        meta['curves'] = curves

    if test_mses is not None:
        meta['test_MSEs'] = test_mses

    return meta


def create_summary_of_regression(prediction, y, metrics='auc', metric_groups=None):
    metrics = coerce_to_tuple_and_check_all_strings(metrics)
    from karma.core.dataframe import DataFrame
    initial_df = DataFrame({'predictions': prediction, 'true_values': y})

    metric_agg_tuple = tuple('{0}(predictions, true_values) as {0}'.format(metric) for metric in metrics)
    agg_tuple = ('# as Count', 'sum(true_values) as CountPos',) + metric_agg_tuple  # only makes sense for binary target
    df_grouped = initial_df.group_by(metric_groups, agg_tuple)

    return df_grouped
