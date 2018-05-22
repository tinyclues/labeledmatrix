from itertools import izip, imap
from multiprocessing.pool import ThreadPool

import numpy as np
from math import ceil

from sklearn.model_selection import StratifiedShuffleSplit

from cyperf.tools import take_indices, logit_inplace
from karma.core.karmacode.utils import RegressionKarmaCodeFormatter

from karma.core.utils import is_iterable, quantile_boundaries
from karma.core.utils.array import is_binary
from karma.learning.matrix_utils import (safe_hstack, number_nonzero, cast_float32,
                                         direct_product, direct_product_dot,
                                         direct_product_dot_transpose,
                                         direct_product_second_moment)
from karma.learning.regression import create_meta_of_regression, create_summary_of_regression
from karma.thread_setter import blas_threads, open_mp_threads

NB_THREADS_MAX = 16
KNOWN_LOGISTIC_METHODS = ['logistic_coefficients', 'logistic_coefficients_and_posteriori']

class VirtualDirectProduct(object):
    """
    We may also want to support scalar case of x or y ?


        X should accept list like this [x, y, z, (x,y), (z, x)]
        tuples stay for direct_product

    """
    def __init__(self, left, right, is_transposed=False):
        assert left.shape[0] == right.shape[0]
        assert left.ndim == 2
        assert right.ndim == 2
        self.is_transposed = is_transposed
        self.left, self.right = cast_float32(left), cast_float32(right)

    @property
    def T(self):
        return VirtualDirectProduct(self.left, self.right, is_transposed=not self.is_transposed)

    @property
    def shape(self):
        shape = (self.left.shape[0], self.left.shape[1] * self.right.shape[1])
        return shape[::-1] if self.is_transposed else shape

    @property
    def dtype(self):
        return np.promote_types(self.left.dtype, self.right.dtype)

    @property
    def nnz(self):
        # it does not work at all case (by sparse x dense should be Ok)
        return number_nonzero(self.left) * number_nonzero(self.right) / max(self.right.shape[0], 1)

    @property
    def ndim(self):
        return 2

    def __getitem__(self, indices):
        if self.is_transposed:
            raise NotImplementedError('VirtualDirectProduct.__getitem__ when is_transposed')
        else:
            return VirtualDirectProduct(self.left[indices], self.right[indices], False)

    def sum(self, axis=None):
        if axis is None:
            return self.sum(axis=1).sum()

        if axis == 1:
            return self.dot(np.ones(self.shape[1], dtype=self.dtype))
        elif axis == 0:
            return self.transpose_dot(np.ones(self.shape[0], dtype=self.dtype))
        else:
            raise NotImplementedError(axis)

    def dot(self, w, power=1):
        if self.is_transposed:
            return direct_product_dot_transpose(self.left, self.right, w, power)
        else:
            return direct_product_dot(self.left, self.right, w, power)

    def transpose_dot(self, w, power=1):
        if self.is_transposed:
            return direct_product_dot(self.left, self.right, w, power)
        else:
            return direct_product_dot_transpose(self.left, self.right, w, power)

    def second_moment(self):
        if self.is_transposed:
            raise NotImplementedError()
        else:
            return direct_product_second_moment(self.left, self.right)

    def materialize(self):
        res = direct_product(self.left, self.right)
        return res.T if self.is_transposed else res

    def __array__(self):
        # to simplify equality tests with numpy
        return np.asarray(self.materialize())


def safe_vdp_cast(x):
    return VirtualDirectProduct(*x) if isinstance(x, tuple) else x


def vdp_materilaze(x):
    return x.materialize() if isinstance(x, VirtualDirectProduct) else x


class BasicVirtualHStack(object):
    def __init__(self, X):
        if isinstance(X, BasicVirtualHStack):
            self.X = X.X
            self.is_block = X.is_block
            if self.is_block:
                self.dims = X.dims
            return

        self.is_block = isinstance(X, (list, tuple))
        if self.is_block:
            if len(X) == 0:
                raise ValueError('Cannot create a VirtualHStack of an empty list')

            X = map(safe_vdp_cast, X)

            integer_entries = []
            common_shape0 = None
            for i in xrange(len(X)):
                if not hasattr(X[i], 'shape') or X[i].shape is None or X[i].shape == ():
                    try:
                        X[i] = int(X[i])
                    except (ValueError, TypeError):
                        raise ValueError('Cannot create a VirtualHStack with {}'.format(type(X[i])))
                    integer_entries.append(i)
                else:
                    shape0 = X[i].shape[0]
                    if common_shape0 is None:
                        common_shape0 = shape0
                    elif shape0 != common_shape0:
                        raise ValueError('Cannot create a VirtualHStack for an array of length {} while other array '
                                         'has length {}'.format(shape0, common_shape0))
            for i in integer_entries:
                X[i] = np.zeros((common_shape0 or 1, X[i]))

            X = [xx if isinstance(xx, VirtualDirectProduct) else cast_float32(xx) for xx in X]
            self.dims = np.cumsum([0] + [x.shape[1] for x in X])
        else:
            X = X if isinstance(X, VirtualDirectProduct) else cast_float32(X)

        self.X = X

    def __getitem__(self, indices):
        if indices is None:
            return self
        if self.is_block:
            return BasicVirtualHStack([x[indices] for x in self.X])
        else:
            return BasicVirtualHStack(self.X[indices])

    @property
    def shape(self):
        if self.is_block:
            return self.X[0].shape[0], self.dims[-1]
        else:
            return self.X.shape

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
                assert (len(res) == self.shape[1])
                return res
            else:
                assert False
        except AssertionError:
            additional_message = ' or {}'.format(len(self.X)) if self.is_block else ''
            param_message = "parameter '{}'".format(param_name) if param_name != '' else 'parameter'
            raise ValueError('{} is invalid: expected float or an array-like of length {}{}, '
                             'got array-like of length {}'.format(param_message, self.shape[1], additional_message,
                                                                  len(arr)))

    def adjust_to_block_dimensions(self, arr, param_name=''):
        return self.split_by_dims(self.adjust_array_to_total_dimension(arr, param_name))

    def materialize(self):
        if self.is_block:
            if len(self.X) == 1:
                return vdp_materilaze(self.X[0])
            else:
                return safe_hstack(map(vdp_materilaze, self.X))
        else:
            return vdp_materilaze(self.X)


class VirtualHStack(BasicVirtualHStack):
    def __init__(self, X, nb_threads=1, nb_inner_threads=None):
        super(VirtualHStack, self).__init__(X)
        self.pool = None
        self.nb_threads = nb_threads
        self.nb_inner_threads = nb_inner_threads
        self._set_parallelism()

    def __repr__(self):
        repr_ = "<VirtualHStack :"
        repr_ += '\n * shape {}'.format(self.shape)
        if self.is_block:
            repr_ += "\n * containing {} blocks".format(len(self.X))
            repr_ += "\n * with dimensions {}".format(np.diff(self.dims))
        else:
            repr_ += "\n * containing single matrix"
        repr_ += "\n *  nb_threads={}, nb_inner_threads={}".format(self.nb_threads, self.nb_inner_threads)
        return unicode(repr_ + ">", encoding="utf-8", errors='replace').encode("utf-8")

    def __del__(self):
        self._close_pool()  # I hope that it works as I expect
        self.X = None

    def _set_parallelism(self):
        if self.is_block:
            self.nb_threads = min(self.nb_threads, len(self.X), NB_THREADS_MAX)
            if self.nb_threads > 1:
                self.pool = ThreadPool(self.nb_threads)

            self.order = np.argsort(map(self._block_priority, self.X))[::-1]
            self.reverse_order = np.argsort(self.order)
        else:
            self.nb_threads = 1

        if self.nb_inner_threads is None:
            if self.nb_threads > 0:
                self.nb_inner_threads = min(NB_THREADS_MAX, max(1, int(2 * 32. / self.nb_threads)))
            else:
                self.nb_inner_threads = NB_THREADS_MAX

    def __getitem__(self, indices):
        if indices is None:
            return self
        return VirtualHStack(super(VirtualHStack, self).__getitem__(indices),
                             self.nb_threads,
                             self.nb_inner_threads)

    def _close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.terminate()
            self.pool = None

    @staticmethod
    def _block_priority(x):
        if isinstance(x, np.ndarray):
            return x.shape[1]
        else:  # KarmaSparse dot is 2 times slow than numpy.dot
            return int(2. * x.nnz / max(x.shape[0], 1))

    def dot(self, w, row_indices=None):
        with blas_threads(self.nb_inner_threads), open_mp_threads(self.nb_inner_threads):
            if self.is_block:
                def _dot(i):
                    x = self.X[i]
                    if row_indices is not None:
                        x = x[row_indices]
                    return x.dot(w[self.dims[i]:self.dims[i + 1]].astype(x.dtype, copy=False))

                if self.pool is not None:
                    return reduce(np.add, self.pool.imap_unordered(_dot, self.order))
                else:
                    return reduce(np.add, imap(_dot, self.order))
            else:
                return self.X.dot(w.astype(self.X.dtype, copy=False))

    def transpose_dot(self, w):
        with blas_threads(self.nb_inner_threads), open_mp_threads(self.nb_inner_threads):
            if self.is_block:
                def _dot(i):
                    x = self.X[i]
                    return x.T.dot(w.astype(x.dtype, copy=False))

                if self.pool is not None:
                    with blas_threads(1):
                        return np.hstack(take_indices(self.pool.map(_dot, self.order), self.reverse_order))
                else:
                    return np.hstack(map(_dot, range(len(self.X))))
            else:
                return self.X.T.dot(w.astype(self.X.dtype, copy=False))


def validate_regression_model(blocks_x, y, cv, method, warmup_key=None, cv_groups=None, cv_n_splits=1, cv_seed=None,
                              **kwargs):
    if not isinstance(cv, CrossValidationWrapper):
        cv = CrossValidationWrapper(cv, y, cv_groups, cv_n_splits, cv_seed)
    cv.validate(blocks_x, y, method, warmup_key, **kwargs)
    return cv


def _prepare_and_check_classes(y, groups):
    if groups is not None:
        assert len(groups) == len(y)
    y = np.asarray(y)
    if is_binary(y):
        classes = y
    else:
        # in case the response is not binary, stratify by quantile
        # at most 10, but never less than 100 lines per quantile
        y_boundaries = quantile_boundaries(y, nb=min(max(len(y) / 100, 1), 10))
        classes = y_boundaries.searchsorted(y)

    if groups is not None:
        groups = np.char.asarray(groups)
        classes = np.char.asarray(classes) + '_' + groups
        unique, counts = np.unique(classes, return_counts=True)

        groups_to_clean = set()
        for _class in unique[counts == 1]:
            groups_to_clean.add(groups[classes == _class][0])
        for group in groups_to_clean:
            classes[groups == group] = group

    _, counts = np.unique(classes, return_counts=True)
    if not np.all(counts > 1):
        raise ValueError("StratifiedShuffleSplit doesn't support classes of size 1")
    return classes


class CrossValidationWrapper(object):
    """Cross-validation.

    General Karma wrapper for the creation of shuffled cross-validation (possibly stratified).

    Parameters
    ----------
    cv : float : Between 0 and 1, proportion of the data to use in each test set.
    y : array : Targets for the classification task.
    groups : array, default None : Values of the stratification variable.
    n_splits : int, default 1 : Number of splits to create.
    seed: int, default None : To use a specific seed.

    Attributes
    ----------
    test_indices : int array : Rows indices used in test sets.
    test_y_hat : float array : Predictions on the different test sets.
    intercepts : list : Intercepts on each fold.
    feat_coefs : list of arrays : List of the feature coefficients on each fold.
    test_fraction : float : Proportion of data in each test set.
    test_size : int : Number of observations in each test set.
    classes : array : Stratified targets.
    n_splits : int : Number of splits.
    seed : int : Random seed used.
    """
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
        self.intercepts, self.feat_coefs = [None] * self.n_splits, [None] * self.n_splits

        from karma.core.dataframe import DataFrame
        self.summary = DataFrame()

    @property
    def cv(self):
        return StratifiedShuffleSplit(self.n_splits, test_size=self.test_fraction, random_state=self.seed)

    @property
    def fold_indices_iter(self):
        for i in xrange(self.n_splits):
            fold_slice = slice(i * self.test_size, (i + 1) * self.test_size)
            yield self.test_indices[fold_slice], self.test_y_hat[fold_slice]

    def validate(self, blocks_x, y, method, warmup_key=None, **kwargs):
        X_stacked = VirtualHStack(blocks_x,
                                  nb_threads=kwargs.get('nb_threads', 1),
                                  nb_inner_threads=kwargs.get('nb_inner_threads'))
        i, j = 0, 0
        compute_gaincurves = kwargs.pop('compute_gaincurves', True)
        compute_summary = kwargs.pop('compute_summary', False)
        metrics = kwargs.pop('metrics', 'auc')
        metric_groups = kwargs.pop('metric_groups', None)
        kc_formatter = kwargs.pop('kc_formatter', None)

        for (train_idx, test_idx) in self.cv.split(self.classes, self.classes):
            train_kwargs = {k: v[train_idx] if isinstance(v, np.ndarray) and v.shape == y.shape else v
                            for k, v in kwargs.items()}
            self.method_output = method(X_stacked[train_idx], y[train_idx], **train_kwargs)
            intercept, betas = self.method_output[1:3]
            self.intercepts[j] = intercept
            self.feat_coefs[j] = betas
            j += 1

            if kc_formatter is None:
                y_hat = np.asarray(X_stacked.dot(np.hstack(betas), row_indices=test_idx) + intercept)
                if method.func_name in KNOWN_LOGISTIC_METHODS:
                    logit_inplace(y_hat)
            else:
                assert isinstance(kc_formatter, RegressionKarmaCodeFormatter)
                kc = kc_formatter.format(self.method_output)
                y_hat = kc.bulk_call(X_stacked[test_idx].X)[0]

            self.test_y_hat[i:i + self.test_size] = y_hat
            self.test_indices[i:i + self.test_size] = test_idx

            if compute_summary and method.func_name in KNOWN_LOGISTIC_METHODS:
                fold_summary = create_summary_of_regression(prediction=y_hat, y=y[test_idx],
                                                            metrics=metrics, metric_groups=metric_groups)
                from karma.macros import squash
                self.summary = squash(self.summary, fold_summary)

            i += self.test_size
            if warmup_key is not None:
                kwargs[warmup_key] = np.hstack(betas + [intercept])
        X_stacked._close_pool()  # Warning should called manually at the exit from class

        if compute_gaincurves:
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
            renaming_dict = {'# train': '#', '# positive train': '# positive'}
            exclude = ['# test', '# positive test']
            cols = [name if name not in renaming_dict else '{} as {}'.format(name, renaming_dict[name])
                    for name in res[col].column_names if name not in exclude]
            res[col] = res[col].copy(*cols)
        return res

    @staticmethod
    def create_cv_from_data_and_params(dataframe, **parameters):
        cv_params_dict = dict()
        cv_params_dict['cv'] = parameters['cv']

        if isinstance(cv_params_dict['cv'], CrossValidationWrapper):
            return dataframe, cv_params_dict['cv']
        else:
            cv_params_dict['n_splits'] = parameters.get('cv_n_splits', 1)
            cv_params_dict['seed'] = parameters.get('seed')

            stratification_col_name = parameters.get('cv_groups')

            if stratification_col_name is None:
                groups = None
            else:
                groups = np.asarray(dataframe[stratification_col_name][:])
                unique, counts = np.unique(groups, return_counts=True)
                if np.any(counts == 1):
                    drop_mask = np.zeros(len(dataframe), dtype=bool)
                    for g in unique[counts == 1]:
                        drop_mask += groups == g
                    kept_indices = np.arange(len(groups))[~drop_mask]
                    dataframe = dataframe[kept_indices]
                    groups = groups[kept_indices]

            y = dataframe[parameters['axis']][:]
            return dataframe, CrossValidationWrapper(y=y, groups=groups, **cv_params_dict)


def check_axis_values(y):
    axis_unique_values = np.unique(y)
    if axis_unique_values.tolist() not in ([0.], [1.], [0., 1.]):
        raise ValueError('Set of values taken by axis, {}, is not a subset of [0, 1]'
                         .format(axis_unique_values))


def calculate_train_test_metrics(dataframe, group_by_col, pred_col, response_col, split_col=None):
    """
    Return a DataFrame with AUC, RMSE/RIG and Calibration metrics
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
    --------------------------------------------------------------------------------------------------------------------------------------
    topic | # Train | # Test | # positive Train | # positive Test | AUC Train | AUC Test | RIG Train | RIG Test | Calib Train | Calib Test
    --------------------------------------------------------------------------------------------------------------------------------------
    0       168       39       96                 22                -0.1117     -0.0053    -0.6616     -0.3572     1.1391        1.03
    1       140       47       76                 25                -0.1439     -0.0945    -0.5192     -0.6478     0.945         1.0744
    2       160       47       76                 22                0.1419      -0.1164    -0.3041     -0.5429     0.9598        0.9852
    3       163       34       78                 20                -0.0042     -0.0929    -0.4236     -0.5821     0.9673        1.0748
    4       169       33       92                 14                -0.026      -0.2707    -0.4532     -0.674      1.0984        0.8872
    >>> calculate_train_test_metrics(df, 'topic', 'pred', 'obs').preview() #doctest: +NORMALIZE_WHITESPACE
    -----------------------------------------------------
    topic | #   | # positive | AUC     | RIG     | Calib
    -----------------------------------------------------
    0       207   118          -0.091    -0.6041   1.117
    1       187   101          -0.117    -0.5515   0.974
    2       207   98           0.0813    -0.3582   0.9654
    3       197   98           -0.0155   -0.4433   0.9875
    4       202   106          -0.0631   -0.4804   1.0649
    >>> calculate_train_test_metrics(df.with_column('z', 'multiply(map(topic, mapping=dict({2: 1}), default=0), obs)'),
    ...                              'topic', 'pred', 'z').preview()  #doctest: +NORMALIZE_WHITESPACE
    ----------------------------------------------------
    topic | #   | # positive | AUC    | RIG     | Calib
    ----------------------------------------------------
    2       207   98           0.0813   -0.3582   0.9654
    """
    is_resp_binary = is_binary(dataframe[response_col][:])
    if is_resp_binary:
        err_agg, err_agg_name = ('relative_information_gain', 'RIG')
    else:
        err_agg, err_agg_name = ('rmse', 'RMSE')

    group_by = (group_by_col, split_col) if split_col is not None else group_by_col
    agg_args = '{}, {}'.format(pred_col, response_col)
    agg_tuple = ('#',
                 'sum({}) as # positive'.format(response_col),
                 'auc({}) as AUC'.format(agg_args),
                 '{}({}) as {}'.format(err_agg, agg_args, err_agg_name),
                 'calibration_ratio({}) as Calib'.format(agg_args))
    metrics = dataframe.group_by(group_by, agg_tuple)
    if is_resp_binary:
        metrics = metrics.where_gt('# positive', 0)
    cols = metrics.column_names[len(group_by):]
    if split_col is not None:
        res_df = dataframe.deduplicate_by(group_by_col, take='first').sort_by(group_by_col).copy(group_by_col)
        metrics_by_label = metrics.split_by(split_col)
        for col in cols:
            for label in sorted(metrics_by_label.keys())[::-1]:
                df = metrics_by_label[label].copy(group_by_col, col)
                _name = df.temporary_column_name()
                df[_name] = df['alias({})'.format(col)]
                res_df.add_relation('rel', df, group_by_col, group_by_col)
                res_df['{} {}'.format(col, label)] = \
                    res_df['round(replace_exceptional(!rel.{}, constant=-1), precision=4)'.format(_name)]
    else:
        res_df = metrics
    return res_df
