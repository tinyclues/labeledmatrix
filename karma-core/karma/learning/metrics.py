import numpy as np
from sklearn.utils import check_array, check_consistent_length
from cyperf.tools import parallel_unique
from scipy.stats import rankdata

__all__ = ['normalized_log_loss_from_prediction', 'auc_from_prediction']


def check_metric_arrays(predicted_values, true_values):
    predicted_values = check_array(predicted_values, ensure_2d=False, dtype=np.float)
    true_values = check_array(true_values, ensure_2d=False, dtype=np.float)
    check_consistent_length(predicted_values, true_values)
    return predicted_values, true_values


def normalized_log_loss_from_prediction(predicted_values, true_values):
    """
    Returns normalized log loss between prediction and observation as it is described in
        https://pdfs.semanticscholar.org/daf9/ed5dc6c6bad5367d7fd8561527da30e9b8dd.pdf
    Args:
        predicted_values: array-like of score prediction values
        true_values: array-like of observed true values
    """
    predicted_values, true_values = check_metric_arrays(predicted_values, true_values)

    unique_true_values = set(parallel_unique(true_values))

    if len(unique_true_values) == 1:
        if unique_true_values == {1.} and np.min(predicted_values) == 1:
            return 0.
        elif unique_true_values == {0.} and np.max(predicted_values) == 0:
            return 0.
        else:
            return np.nan  # we prefer to return exceptional value

    if unique_true_values != {0., 1.}:
        raise ValueError('Normalized logloss can be computed only for binary true_values')

    idx_bad_predictions = np.bitwise_or(predicted_values == 0, predicted_values == 1)
    if np.sum(np.bitwise_xor(true_values[idx_bad_predictions].astype(np.bool),
                             predicted_values[idx_bad_predictions].astype(np.bool))) > 0:
        return np.nan

    idx = np.bitwise_not(idx_bad_predictions)
    true_values = true_values[idx]
    predicted_values = predicted_values[idx]

    global_p = np.mean(true_values)
    mean_entropy = np.mean(true_values * np.log(predicted_values) + (1 - true_values) * np.log(1 - predicted_values))
    global_mean_entropy = global_p * np.log(global_p) + (1 - global_p) * np.log(1 - global_p)

    return mean_entropy / global_mean_entropy


def auc_from_prediction(predicted_values, true_values):
    """
    >>> from karma.core.dataframe import DataFrame
    >>> df = DataFrame({'name': ['a', 'a', 'b', 'a', 'b', 'b'],
    ...                 'pred': [0.1, 0.2, 0.1, 0.4, 0.6, 0.2],
    ...                 'real': [0, 1, 0, 1, 0, 1]})
    >>> df.group_by('name', 'auc(pred, real)').preview() #doctest: +NORMALIZE_WHITESPACE
    ----------------------
    name | auc(pred, real)
    ----------------------
    a      1.0
    b      0.0
    """
    predicted_values, true_values = check_metric_arrays(predicted_values, true_values)

    unique_true_values = set(parallel_unique(true_values))

    if len(unique_true_values) == 1:
        if unique_true_values == {1.} or unique_true_values == {0.}:
            return 1.
        else:
            return np.nan  # we prefer to return exceptional value

    if unique_true_values != {0., 1.}:
        raise ValueError('Non binary AUC is not implemented yet.')

    pred_pos = predicted_values[true_values == 1]
    n_pos = len(pred_pos)

    pred_neg = predicted_values[true_values == 0]
    n_neg = len(pred_neg)

    rankings = rankdata(np.hstack([pred_pos, pred_neg]))

    unnormalized_auc = np.sum(rankings[:n_pos]) - (n_pos * (n_pos + 1)) / 2.
    auc_value = 2 * (unnormalized_auc / (n_pos * n_neg)) - 1
    return np.round(auc_value, decimals=4)


def root_mean_squared_error(predicted_values, true_values):
    predicted_values, true_values = check_metric_arrays(predicted_values, true_values)
    return np.sqrt(np.mean((predicted_values - true_values) ** 2))


def absolute_mean_error(predicted_values, true_values):
    predicted_values, true_values = check_metric_arrays(predicted_values, true_values)
    return np.mean(np.abs(predicted_values - true_values))
