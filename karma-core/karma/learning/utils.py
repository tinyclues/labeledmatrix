from itertools import izip
from multiprocessing.pool import ThreadPool

import numpy as np
from math import ceil

from sklearn.model_selection import StratifiedShuffleSplit

from cyperf.tools import take_indices, logit_inplace

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

            self.nb_threads = min(nb_threads, len(X), 16)
            self.pool = ThreadPool(self.nb_threads)
            self.nb_inner_threads = min(16, max(1, int(2 * 32. / self.nb_threads)))
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


def validate_regression_model(blocks_x, y, cv, method, warmup_key=None, cv_groups=None, cv_n_splits=1, cv_seed=None,
                              **kwargs):
    if not isinstance(cv, CrossValidationWrapper):
        cv = CrossValidationWrapper(cv, y, cv_groups, cv_n_splits, cv_seed)
    cv.validate(blocks_x, y, method, warmup_key, **kwargs)
    return cv


def _prepare_and_check_classes(y, groups):
    if is_binary(y):
        classes = y
        if groups is not None:
            groups = np.char.asarray(groups)
            classes = np.char.asarray(classes) + '_' + groups
            unique, counts = np.unique(classes, return_counts=True)

            groups_to_clean = set()
            for _class in unique[counts == 1]:
                groups_to_clean.add(groups[classes == _class][0])
            for group in groups_to_clean:
                classes[groups == group] = group
    else:
        classes = groups if groups is not None else np.zeros(len(y))

    _, counts = np.unique(classes, return_counts=True)
    if not np.all(counts > 1):
        raise ValueError("StratifiedShuffleSplit doesn't support classes of size 1")
    return classes


class CrossValidationWrapper(object):
    method_output = None
    meta = None

    def __init__(self, cv, y, groups=None, n_splits=1, seed=None):
        if not (isinstance(cv, float) and 0 < cv < 1):
            raise ValueError('CvIterator only support cv to be a float in (0, 1)')
        self.test_fraction = cv
        self.test_size = int(ceil(cv * len(y)))  # sklearn/model_selection/_split.py l.1379

        self.classes = _prepare_and_check_classes(y, groups)

        self.n_splits = n_splits
        self.seed = seed if seed is not None else len(y)

        self.test_indices = np.zeros(self.test_size * self.n_splits, dtype=int)
        self.test_y_hat = np.zeros(self.test_size * self.n_splits, dtype=np.float64)

    def validate(self, blocks_x, y, method, warmup_key=None, **kwargs):
        cv = StratifiedShuffleSplit(self.n_splits, test_size=self.test_fraction, random_state=self.seed)

        X_stacked = VirtualHStack(blocks_x, nb_threads=kwargs.get('nb_threads', 1))
        i = 0
        for (train_idx, test_idx) in cv.split(self.classes, self.classes):
            train_kwargs = {k: v[train_idx] if isinstance(v, np.ndarray) and v.shape == y.shape else v
                            for k, v in kwargs.items()}
            self.method_output = method(X_stacked[train_idx], y[train_idx], **train_kwargs)
            intercept, betas = self.method_output[1:3]
            self.test_y_hat[i:i + self.test_size] = np.asarray(
                X_stacked.dot(np.hstack(betas), row_indices=test_idx) + intercept)
            self.test_indices[i:i + self.test_size] = test_idx
            i += self.test_size
            if warmup_key is not None:
                kwargs[warmup_key] = np.hstack(betas + [intercept])
        X_stacked._close_pool()  # Warning should called manually at the exit from class

        if method.func_name.startswith('logistic'):
            logit_inplace(self.test_y_hat)

        self.meta = create_meta_of_regression(self.test_y_hat, y[self.test_indices], with_guess=False)

    def calculate_train_test_metrics(self, trained_dataframe, group_by_cols, pred_col, response_col):
        from karma.core.column import create_column_from_data
        from karma.macros import squash

        test_dataframe = trained_dataframe.copy(response_col, *group_by_cols)[self.test_indices]
        test_dataframe[pred_col] = create_column_from_data(self.test_y_hat)

        dataframe = squash({'train': trained_dataframe.copy(response_col, pred_col, *group_by_cols),
                            'test': test_dataframe}, lazy=True, label='label')
        res = {}
        for col in group_by_cols:
            res[col] = calculate_train_test_metrics(dataframe, col, pred_col, response_col, split_col='label')
        return res


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
    ------------------------------------------------------------------------------------
    topic | #   | AUC Train | AUC Test | NLL Train | NLL Test | Calib Train | Calib Test
    ------------------------------------------------------------------------------------
    0       207   -0.1117     -0.0053    1.6616      1.3572     1.1391        1.03
    1       187   -0.1439     -0.0945    1.5192      1.6478     0.945         1.0744
    2       207   0.1419      -0.1164    1.3041      1.5429     0.9598        0.9852
    3       197   -0.0042     -0.0929    1.4236      1.5821     0.9673        1.0748
    4       202   -0.026      -0.2707    1.4532      1.674      1.0984        0.8872
    """

    if is_binary(dataframe[response_col][:]):
        err_agg, err_agg_name = ('normalized_log_loss', 'NLL')
    else:
        err_agg, err_agg_name = ('rmse', 'RMSE')

    group_by = (group_by_col, split_col) if split_col is not None else group_by_col
    agg_args = '{}, {}'.format(pred_col, response_col)
    agg_tuple = ('#',
                 'auc({}) as AUC'.format(agg_args),
                 '{}({}) as {}'.format(err_agg, agg_args, err_agg_name),
                 'calibration_ratio({}) as Calib'.format(agg_args))
    metrics = dataframe.group_by(group_by, agg_tuple)
    if split_col is not None:
        res_df = dataframe.group_by(group_by_col, '#')
        metrics_by_label = metrics.split_by(split_col)
        for col in ['AUC', err_agg_name, 'Calib']:
            for label in sorted(metrics_by_label.keys())[::-1]:
                df = metrics_by_label[label]
                res_df.add_relation('rel', df, group_by_col, group_by_col)
                res_df['{} {}'.format(col, label)] = \
                    res_df['round(translate(translate(!rel.{}, mapping={{RelationalMissing: -1}}), '
                           'mapping={{np.inf: -1}}), precision=4)'.format(col)]
    else:
        res_df = metrics
    return res_df
