#
# Copyright tinyclues, All rights reserved
#
from functools import wraps

import numpy as np
from lightfm import LightFM
from sklearn.linear_model import Ridge

from cyperf.matrix.karma_sparse import is_karmasparse

from karma.core.utils.utils import coerce_to_tuple_and_check_all_strings
from karma.core.curve import compute_mean_curve, gain_curve_from_prediction
from karma.core.utils.array import is_binary
from karma.learning.matrix_utils import to_scipy_sparse

__all__ = ['regression_on_blocks', 'linear_regression_coefficients', 'sklearn_regression_model',
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


@regression_on_blocks
def sklearn_regression_model(X, y, regression_class, classifier=False, **kwargs):
    if 'max_features' in kwargs:
        kwargs['max_features'] = min(X.shape[1], kwargs['max_features'])

    if classifier:
        y = y * 2 - 1

    if is_karmasparse(X):
        X = X.to_scipy_sparse(copy=False)

    clf = regression_class(random_state=X.shape[0], **kwargs)
    clf.fit(X, y)

    y_hat = clf.predict_proba(X) if classifier else clf.predict(X)

    return y_hat, clf


def lightfm_regression_model(user_features, item_features, interactions, sample_weight=None, **kwargs):
    # user_features and item_features must respect order of users and items from interactions matrix
    user_features = to_scipy_sparse(user_features)
    item_features = to_scipy_sparse(item_features)
    interactions = to_scipy_sparse(interactions)

    if sample_weight is not None:
        sample_weight = to_scipy_sparse(sample_weight)

    clf = LightFM(no_components=kwargs['no_components'],
                  k=kwargs.get('k', 5),
                  n=kwargs.get('n', 10),
                  learning_schedule=kwargs.get('learning_schedule', 'adagrad'),
                  loss=kwargs.get('loss', 'logistic'),
                  learning_rate=kwargs.get('learning_rate', 0.05),
                  rho=kwargs.get('rho', 0.95),
                  epsilon=kwargs.get('epsilon', 1e-6),
                  item_alpha=kwargs.get('item_alpha', 0.0),
                  user_alpha=kwargs.get('user_alpha', 0.0),
                  max_sampled=kwargs.get('max_sampled', 10),
                  random_state=user_features.shape[0])
    clf.fit(interactions,
            user_features=user_features, item_features=item_features,
            sample_weight=sample_weight,
            epochs=kwargs.get('epochs', 1),
            num_threads=kwargs.get('num_threads', 1),
            verbose=kwargs.get('verbose', False))
    # We need to replace user and items labels by integers
    user_ids, item_ids = interactions.nonzero()
    y = np.ravel(interactions[user_ids, item_ids])

    y_linear = clf.predict(user_ids=user_ids, item_ids=item_ids,
                           item_features=item_features, user_features=user_features,
                           num_threads=kwargs.get('num_threads', 1))

    return y, y_linear, clf


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
