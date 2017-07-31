from itertools import izip
from multiprocessing.pool import ThreadPool

import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from cyperf.tools import take_indices, logit

from karma.core.utils import is_iterable
from karma.core.utils.array import is_binary
from karma.learning.matrix_utils import safe_hstack
from karma.learning.regression import create_meta_of_regression
from karma.thread_setter import blas_threads, open_mp_threads


class VirtualHStack(object):

    def __init__(self, X, nb_threads=1):
        self.is_block = isinstance(X, (list, tuple))
        if self.is_block:
            assert len(X) > 0
            assert all(X[0].shape[0] == x.shape[0] for x in X[1:])
            self.dims = np.cumsum([0] + [x.shape[1] for x in X])

            self.nb_threads = min(nb_threads, len(X), 32)
            self.pool = ThreadPool(self.nb_threads)
            self.nb_inner_threads = max(1, int(2 * 32. / self.nb_threads))
            self.order = np.argsort(map(self._block_priority, X))[::-1]
            self.reverse_order = np.argsort(self.order)
        else:
            self.pool = None
            self.nb_threads = 1
        self.X = X

    # def __del__(self):  # It does not work as I expect
    #     self._close_pool()
    #     self.X = None

    def __getitem__(self, indices):
        if indices is None:
            return self
        if self.is_block:
            return VirtualHStack([x[indices] for x in self.X], self.nb_threads)
        else:
            return VirtualHStack(self.X[indices])

    def _close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.terminate()

    @staticmethod
    def _block_priority(x):
        if isinstance(x, np.ndarray):
            return x.shape[1]
        else:  # KarmaSparse dot is 2 times slow than numpy.dot
            return int(2. * x.nnz / max(x.shape[0], 1))

    @property
    def shape(self):
        if self.is_block:
            return self.X[0].shape[0], self.dims[-1]
        else:
            return self.X.shape

    def dot(self, w, row_indices=None):
        if self.is_block:
            def _dot(i):
                x = self.X[i]
                if row_indices is not None:
                    x = x[row_indices]
                return x.dot(w[self.dims[i]:self.dims[i + 1]].astype(x.dtype, copy=False))

            with blas_threads(self.nb_inner_threads), open_mp_threads(self.nb_inner_threads):
                return reduce(np.add, self.pool.imap_unordered(_dot, self.order))
        else:
            return self[row_indices].X.dot(w.astype(self.X.dtype, copy=False))

    def transpose_dot(self, w):
        if self.is_block:
            def _dot(i):
                x = self.X[i]
                return x.T.dot(w.astype(x.dtype, copy=False))

            with blas_threads(self.nb_inner_threads), open_mp_threads(self.nb_inner_threads):
                return np.hstack(take_indices(self.pool.map(_dot, self.order), self.reverse_order))
        else:
            return self.X.T.dot(w.astype(self.X.dtype, copy=False))

    def split_by_dims(self, w):
        assert w.shape[0] == self.shape[1]
        if self.is_block:
            return [w[self.dims[i]:self.dims[i + 1]] for i in xrange(len(self.X))]
        else:
            return w

    def adjust_array_to_total_dimension(self, arr, param_name=''):
        try:
            if not is_iterable(arr):
                return np.full(self.shape[1], arr)
            elif len(arr) == self.shape[1]:
                return np.asarray(arr)
            elif self.is_block and len(arr) == len(self.X):
                exp_arr = [np.full(x.shape[1], elt) if not is_iterable(elt) else elt for elt, x in izip(arr, self.X)]
                res = np.concatenate(exp_arr)
                assert(len(res) == self.shape[1])
                return res
            else:
                assert False
        except AssertionError:
            additional_message = ' or {}'.format(len(self.X)) if self.is_block else ''
            param_message = "parameter '{}'".format(param_name) if param_name != '' else 'parameter'
            raise ValueError('{} is invalid: expected float or an array-like of length {}{}, '
                             'got array-like of length {}'.format(param_message, self.shape[1], additional_message,
                                                                  len(arr)))

    def flatten_hstack(self):
        if self.is_block:
            if len(self.X) == 1:
                return self.X[0]
            else:
                return safe_hstack(self.X)
        else:
            return self.X


def validate_regression_model(blocks_x, y, cv, method, **kwargs):
    """
    Evaluates model constructed in method on a test set given by cv which can be
        * float in (0, 1) -> new train/test split will be generated
        * ShuffleSplit or StratifiedShuffleSplit object -> train/test split will be obtained by its split method
        * tuple of size two corresponding to train_idx, test_idx
    """
    train_idx, test_idx = get_indices_from_cv(cv, y)
    X_stacked = VirtualHStack(blocks_x, nb_threads=kwargs.get('nb_threads', 1))
    method_output = method(X_stacked[train_idx], y[train_idx], **kwargs)
    intercept, betas = method_output[1:3]
    y_test_hat = X_stacked.dot(np.hstack(betas), row_indices=test_idx) + intercept
    if method.func_name.startswith('logistic'):
        y_test_hat = logit(np.asarray(y_test_hat))
    X_stacked._close_pool()  # Warning should called manually at the exit from class
    metas = create_meta_of_regression(y_test_hat, y[test_idx], with_guess=False)
    test_curve = metas['curves']
    test_mse = metas['train_MSE']

    return (test_curve, test_mse) + method_output


def get_indices_from_cv(cv, y, groups=None, seed=None):
    """
    Return a split of a range(len(y)) into train and test in respect to `cv` parameter
    Args:
        cv: one of:
            * test fraction float in (0, 1)
            * ShuffleSplit or StratifiedShuffleSplit object
            * tuple of size two (in this case method does nothing)
        y: target variable values
        groups: groups to stratify a sample
        seed: seed for the stratified shuffle split
    """
    if isinstance(cv, tuple) and len(cv) == 2:
        return cv

    if isinstance(cv, float) and 0 < cv < 1:
        cv = StratifiedShuffleSplit(n_splits=1, test_size=cv,
                                    random_state=seed if seed is not None else len(y))

    if isinstance(cv, StratifiedShuffleSplit):
        classes = y if is_binary(y) else np.zeros(len(y))
        if groups is not None:
            classes = np.char.asarray(classes) + '_' + np.char.asarray(groups)

        return next(cv.split(y, classes))
    elif isinstance(cv, ShuffleSplit):
        return cv.split(y)
    else:
        raise ValueError('Unknown cv type: {} must be float in (0, 1) or '
                         'ShuffleSplit object or tuple (train_idx, test_idx)'.format(cv))


def check_axis_values(y):
    axis_unique_values = np.unique(y)
    if axis_unique_values.tolist() not in ([0.], [1.], [0., 1.]):
        raise ValueError('Set of values taken by axis, {}, is not a subset of [0, 1]'
                         .format(axis_unique_values))


def calculate_train_test_metrics(dataframe, group_by_col, pred_col, response_col, split_col=None):
    """
    Return a DataFrame with AUC, RMSE/NLL and Calibration metrics
    Args:
        dataframe: input DataFrame with data samples
        group_by_col: group of events for which we want to calculate metrics (example: 'topic_id', 'universe')
        pred_col: prediction column
        response_col: response column
        split_col: columns to split groups into subgroup and to pivot in respect to them (example: Train/Test 'label')
    >>> from karma.core.dataframe import DataFrame
    >>> from karma.core.utils.utils import use_seed
    >>> with use_seed(1515):
    ...     df = DataFrame({'topic': np.random.randint(0, 5, 1000), 'pred': np.random.rand(1000),
    ...         'obs': np.random.randint(0, 2, 1000), 'label': ['Train'] * 800 + ['Test'] * 200})
    >>> calculate_train_test_metrics(df, 'topic', 'pred', 'obs', 'label').preview() #doctest: +NORMALIZE_WHITESPACE
    ------------------------------------------------------------------------------------------
    topic | AUC Train | AUC Test | NLL Train | NLL Test | Calibration Train | Calibration Test
    ------------------------------------------------------------------------------------------
    0       -0.1117     -0.0053    1.6616      1.3572     1.1391              1.03
    1       -0.1439     -0.0945    1.5192      1.6478     0.945               1.0744
    2       0.1419      -0.1164    1.3041      1.5429     0.9598              0.9852
    3       -0.0042     -0.0929    1.4236      1.5821     0.9673              1.0748
    4       -0.026      -0.2707    1.4532      1.674      1.0984              0.8872
    """

    if is_binary(dataframe[response_col][:]):
        err_agg, err_agg_name = ('normalized_log_loss', 'NLL')
    else:
        err_agg, err_agg_name = ('rmse', 'RMSE')

    group_by = (group_by_col, split_col) if split_col is not None else group_by_col
    agg_args = '{}, {}'.format(pred_col, response_col)
    agg_tuple = ('auc({}) as AUC'.format(agg_args),
                 '{}({}) as {}'.format(err_agg, agg_args, err_agg_name),
                 'calibration_ratio({}) as Calibration'.format(agg_args))
    metrics = dataframe.group_by(group_by, agg_tuple)
    if split_col is not None:
        res_df = metrics.copy(group_by_col).deduplicate_by(group_by_col).sort_by(group_by_col)
        metrics_by_label = metrics.split_by(split_col)
        for col in ['AUC', err_agg_name, 'Calibration']:
            for label in sorted(metrics_by_label.keys())[::-1]:
                df = metrics_by_label[label]
                res_df.add_relation('rel', df, group_by_col, group_by_col)
                res_df['{} {}'.format(col, label)] = \
                    res_df['round(translate(translate(!rel.{}, mapping={{RelationalMissing: -1}}), '
                           'mapping={{np.inf: -1}}), precision=4)'.format(col)]
    else:
        res_df = metrics
    return res_df
