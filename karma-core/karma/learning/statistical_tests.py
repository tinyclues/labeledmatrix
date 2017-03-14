import numpy as np
from scipy.stats import norm, ttest_ind_from_stats


def combine_statistics(N1, N2, mean1, mean2, std1, std2):
    """
    Combine basic statistics.
    Source: https://en.wikipedia.org/wiki/Standard_deviation#Combining_standard_deviations

    >>> samples = 5. + 2. * np.random.randn(100)
    >>> samples1, samples2 = samples[:40], samples[40:]
    >>> N, mean, std = 100, samples.mean(), samples.std()
    >>> N1, mean1, std1 = 40, samples1.mean(), samples1.std()
    >>> N2, mean2, std2 = 60, samples2.mean(), samples2.std()
    >>> new_N, new_mean, new_std = combine_statistics(N1, N2, mean1, mean2, std1, std2)
    >>> np.abs(new_N - N) < 1e-10
    True
    >>> np.abs(new_mean - mean) < 1e-10
    True
    >>> np.abs(new_std - std) < 1e-10
    True

    >>> samples = 5. + 2. * np.random.randn(100, 5)
    >>> samples1, samples2 = samples[:40], samples[40:]
    >>> N, mean, std = 100 * np.ones(5, dtype='int'), samples.mean(axis=0), samples.std(axis=0)
    >>> N1, mean1, std1 = 40 * np.ones(5, dtype='int'), samples1.mean(axis=0), samples1.std(axis=0)
    >>> N2, mean2, std2 = 60 * np.ones(5, dtype='int'), samples2.mean(axis=0), samples2.std(axis=0)
    >>> new_N, new_mean, new_std = combine_statistics(N1, N2, mean1, mean2, std1, std2)
    >>> new_N.shape[0] == 5
    True
    >>> new_mean.shape[0] == 5
    True
    >>> new_std.shape[0] == 5
    True
    >>> np.all(np.abs(new_N - N) < 1e-10)
    True
    >>> np.all(np.abs(new_mean - mean) < 1e-10)
    True
    >>> np.all(np.abs(new_std - std) < 1e-10)
    True
    """
    new_N = N1 + N2
    new_mean = (N1 * mean1 + N2 * mean2) / new_N
    new_std = np.sqrt((N1 * std1 ** 2 + N2 * std2 ** 2) / new_N +
                      (N1 * N2 * (mean1 - mean2) ** 2) / new_N ** 2)
    return new_N, new_mean, new_std


class StatsObject(object):

    def __init__(self, N, mean, std):
        """
        >>> stats = StatsObject(1, 2.0, 3.0)
        >>> stats.N, stats.mean, stats.std
        (array([1]), array([ 2.]), array([ 3.]))
        >>> stats = StatsObject(1.0, 2.0, 3.0)
        >>> stats.N, stats.mean, stats.std
        (array([1]), array([ 2.]), array([ 3.]))
        >>> stats = StatsObject(1.5, 2.0, 3.0) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: N should be an array or a scalar of type int or have integer value(s)

        >>> N, mean, std = np.array([1, 1, 1]), np.array([1., 1., 1.]), np.array([1., 1., 1.])
        >>> stats = StatsObject(N, mean, std)
        >>> stats.N, stats.mean, stats.std # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        (array([1, 1, 1]), array([ 1.,  1.,  1.]), array([ 1.,  1.,  1.]))
        >>> stats = StatsObject(N.astype('float'), mean, std)
        >>> stats.N, stats.mean, stats.std # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        (array([1, 1, 1]), array([ 1.,  1.,  1.]), array([ 1.,  1.,  1.]))
        >>> stats = StatsObject(np.array([1., 1.1, 1.]), mean, std)
        Traceback (most recent call last):
        ...
        ValueError: N should be an array or a scalar of type int or have integer value(s)
        """
        n = np.atleast_1d(np.asarray(N, dtype='int'))
        if not np.all(n == N):
            raise ValueError("N should be an array or a scalar of type int or have integer value(s)")
        self.N = n
        self.mean = np.atleast_1d(mean)
        self.std = np.atleast_1d(std)
        # self.stats = np.atleast_1d(np.rec.fromarrays((n, mean, std), names=('N', 'mean', 'std')))


    def __add__(self, other):
        """
        >>> stats1 = StatsObject(2, 1., 1.)
        >>> stats2 = StatsObject(2, 1., 1.)
        >>> stats_sum = stats1 + stats2
        >>> stats_sum.N, stats_sum.mean, stats_sum.std
        (array([4]), array([ 1.]), array([ 1.]))
        >>> stats2 = StatsObject(0, 1., 1.)
        >>> stats_sum = stats1 + stats2
        >>> stats_sum.N, stats_sum.mean, stats_sum.std
        (array([2]), array([ 1.]), array([ 1.]))
        >>> stats1 = StatsObject(0, 1., 15.)
        >>> stats_sum = stats1 + stats2
        >>> stats_sum.N, stats_sum.mean, stats_sum.std
        (array([0]), array([ 1.]), array([ 15.]))
        >>> stats1 + 5. # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for +: 'StatsObject' and 'float'
        """
        if not isinstance(other, StatsObject):
            raise TypeError("unsupported operand type(s) for +: 'StatsObject' and '{}'".format(type(other).__name__))
        if np.all(self.N == 0) and np.all(other.N == 0):
            return StatsObject(self.N, self.mean, self.std)
        N1, N2 = self.N, other.N
        mean1, mean2 = self.mean, other.mean
        std1, std2 = self.std, other.std
        new_N, new_mean, new_std = combine_statistics(N1, N2, mean1, mean2, std1, std2)
        return StatsObject(new_N, new_mean, new_std)

    def __iadd__(self, other):
        """
        >>> stats1 = StatsObject(2, 1., 1.)
        >>> stats2 = StatsObject(2, 1., 1.)
        >>> stats1 += stats2
        >>> stats1.N, stats1.mean, stats1.std
        (array([4]), array([ 1.]), array([ 1.]))
        >>> stats1 = StatsObject(2, 1., 1.)
        >>> stats2 = StatsObject(0, 1., 1.)
        >>> stats1 += stats2
        >>> stats1.N, stats1.mean, stats1.std
        (array([2]), array([ 1.]), array([ 1.]))
        >>> stats1 = StatsObject(0, 1., 15.)
        >>> stats1 += stats2
        >>> stats1.N, stats1.mean, stats1.std
        (array([0]), array([ 1.]), array([ 15.]))
        >>> stats1 += 5. # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for +: 'StatsObject' and 'float'
        """
        if not isinstance(other, StatsObject):
            raise TypeError("unsupported operand type(s) for +: 'StatsObject' and '{}'".format(type(other).__name__))
        if other.N == 0:
            return self
        N1, N2 = self.N, other.N
        mean1, mean2 = self.mean, other.mean
        std1, std2 = self.std, other.std
        self.N[:], self.mean[:], self.std[:] = combine_statistics(N1, N2, mean1, mean2, std1, std2)
        return self

    def __eq__(self, other):
        """
        >>> stats1 = StatsObject(2, 1., 1.)
        >>> stats2 = StatsObject(2, 1., 1.)
        >>> stats1 == 5.
        False
        >>> stats1 == stats2
        True
        >>> stats1 == StatsObject(1, 1., 1.)
        False
        >>> stats1 == StatsObject(2, 2., 1.)
        False
        >>> stats1 == StatsObject(2, 1., 2.)
        False
        """
        if not isinstance(other, StatsObject):
            return False
        return np.all(self.N == other.N) and np.allclose(self.mean, other.mean) and np.allclose(self.std, other.std)

    def __repr__(self):
        """
        >>> StatsObject(2, 1., 1.)
        Stats(N=[2], mean=[ 1.], std=[ 1.])
        >>> StatsObject([2,2], [1.,1.], [1.,2.])
        Stats(N=[2 2], mean=[ 1.  1.], std=[ 1.  2.])
        """
        template = 'Stats(N={N}, mean={mean}, std={std})'
        return template.format(N=self.N, mean=self.mean, std=self.std)


class ControlGroupStatsObject(object):

    def __init__(self, stats_ncg, stats_cg):
        if not (isinstance(stats_ncg, StatsObject) and isinstance(stats_cg, StatsObject)):
            raise TypeError('ControlGroupStatsObject only supports StatsObject as inputs')
        self.ncg = stats_ncg
        self.cg = stats_cg

    def __add__(self, other):
        """
        >>> stats_ncg1 = StatsObject(2, 2., 2.)
        >>> stats_cg1 = StatsObject(1, 1., 1.)
        >>> stats_ncg_sum = stats_ncg1 + stats_ncg1
        >>> stats_cg_sum = stats_cg1 + stats_cg1
        >>> cg_stats1 = ControlGroupStatsObject(stats_ncg1, stats_cg1)
        >>> cg_stats_sum = cg_stats1 + cg_stats1
        >>> cg_stats_sum.ncg == stats_ncg_sum
        True
        >>> cg_stats_sum.cg == stats_cg_sum
        True

        >>> stats_cg2 = StatsObject(3, 3., 3.)
        >>> cg_stats2 = ControlGroupStatsObject(stats_ncg1, stats_cg2)
        >>> stats_cg_sum2 = stats_cg1 + stats_cg2
        >>> cg_stats_sum = cg_stats1 + cg_stats2
        >>> cg_stats_sum.ncg == stats_ncg_sum
        True
        >>> cg_stats_sum.cg == stats_cg_sum2
        True
        """
        if not isinstance(other, ControlGroupStatsObject):
            raise TypeError('cannot add objects of types Stats and {}'.format(type(other)))
        return ControlGroupStatsObject(self.ncg + other.ncg, self.cg + other.cg)

    def __iadd__(self, other):
        """
        >>> stats_ncg1 = StatsObject(2, 2., 2.)
        >>> stats_cg1 = StatsObject(1, 1., 1.)
        >>> stats_ncg_sum = stats_ncg1 + stats_ncg1
        >>> stats_cg_sum = stats_cg1 + stats_cg1
        >>> cg_stats1 = ControlGroupStatsObject(stats_ncg1, stats_cg1)
        >>> cg_stats2 = ControlGroupStatsObject(stats_ncg1, stats_cg1)
        >>> cg_stats1 += cg_stats2
        >>> cg_stats1.ncg == stats_ncg_sum
        True
        >>> cg_stats1.cg == stats_cg_sum
        True

        >>> stats_cg2 = StatsObject(3, 3., 3.)
        >>> cg_stats1 = ControlGroupStatsObject(stats_ncg1, stats_cg1)
        >>> cg_stats2 = ControlGroupStatsObject(stats_ncg1, stats_cg2)
        >>> stats_cg_sum2 = stats_cg1 + stats_cg2
        >>> cg_stats1 += cg_stats2
        >>> cg_stats1.ncg == stats_ncg_sum
        True
        >>> cg_stats1.cg == stats_cg_sum2
        True
        """
        if not isinstance(other, ControlGroupStatsObject):
            raise TypeError('cannot add objects of types Stats and {}'.format(type(other)))
        ncg = self.ncg + other.ncg
        cg = self.cg + other.cg
        self.ncg = ncg
        self.cg  = cg
        return self

    def __eq__(self, other):
        """
        >>> cg_stats1 = ControlGroupStatsObject(StatsObject(1, 1., 1.), StatsObject(2, 2., 2.))
        >>> cg_stats2 = ControlGroupStatsObject(StatsObject(1, 1., 1.), StatsObject(2, 2., 2.))
        >>> cg_stats1 == 5.
        False
        >>> cg_stats1 == cg_stats2
        True
        >>> cg_stats1 == ControlGroupStatsObject(StatsObject(2, 1., 1.), StatsObject(2, 2., 2.))
        False
        >>> cg_stats1 == ControlGroupStatsObject(StatsObject(1, 2., 1.), StatsObject(2, 2., 2.))
        False
        >>> cg_stats1 == ControlGroupStatsObject(StatsObject(1, 1., 2.), StatsObject(2, 2., 2.))
        False
        >>> cg_stats1 == ControlGroupStatsObject(StatsObject(1, 1., 1.), StatsObject(1, 2., 2.))
        False
        >>> cg_stats1 == ControlGroupStatsObject(StatsObject(1, 1., 1.), StatsObject(2, 1., 2.))
        False
        >>> cg_stats1 == ControlGroupStatsObject(StatsObject(1, 1., 1.), StatsObject(2, 2., 1.))
        False
        """
        if not isinstance(other, ControlGroupStatsObject):
            return False
        return self.ncg == other.ncg and self.cg == other.cg

    def __repr__(self):
        """
        >>> ncg_stats = StatsObject(2, 1., 1.)
        >>> cg_stats = StatsObject(3, 2., 2.)
        >>> ControlGroupStatsObject(ncg_stats, cg_stats)
        ControlGroupStats(nonCG: N=[2], mean=[ 1.], std=[ 1.]; CG: N=[3], mean=[ 2.], std=[ 2.])
        """
        template = 'ControlGroupStats(nonCG: N={N_ncg}, mean={mean_ncg}, std={std_ncg}; ' \
                                        'CG: N={N_cg}, mean={mean_cg}, std={std_cg})'
        return template.format(N_ncg=self.ncg.N, mean_ncg=self.ncg.mean, std_ncg=self.ncg.std,
                               N_cg=self.cg.N, mean_cg=self.cg.mean, std_cg=self.cg.std)



def minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1):
    """
    Check if the difference between the two means is significative.
    alpha: type I error parameter - probability that we wrongly reject the null hypothesis
           (no difference between the two means)
    beta: type II error parameter - the probability that we we wrongly accept the null hypothesis
    If the test returns True, it means that we have less than alpha probability to reject the null hypothesis
    even though it is true and less than beta probability to accept the null hypothesis even though it is false.
    Source: http://www.vanbelle.org/chapters/webchapter2.pdf

    >>> stats_A = StatsObject(100, 0., 1.)
    >>> stats_B = StatsObject(50, 100., 1.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)[0] # very separated
    True
    >>> stats_A = StatsObject(100, 100., 1.) # 100 samples from a gaussian with mean 0 and variance 1
    >>> stats_B = StatsObject(50, 0., 1.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)[0] # separated, inv
    True
    >>> stats_A = StatsObject(100, 0., 1.)
    >>> stats_B = StatsObject(50, 1., 1.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)[0] # separated
    True
    >>> stats_A = StatsObject(1000, 0., 1.)
    >>> stats_B = StatsObject(500, 0.178, 1.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.05, beta=0.1)[0] # barely sep
    True
    >>> stats_A = StatsObject(1000, 0., 1.)
    >>> stats_B = StatsObject(500, 0.176, 1.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.05, beta=0.1)[0] # barely not sep
    False
    >>> stats_A = StatsObject(100, 0., 1.)
    >>> stats_B = StatsObject(50, 0.5, 50.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)[0] # not separated
    False
    >>> stats_A = StatsObject(50, 0.5, 50.)
    >>> stats_B = StatsObject(100, 0., 1.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)[0] # not sep, switch
    False
    >>> stats_A = StatsObject(100, 0.5, 1.) # almost same statistics
    >>> stats_B = StatsObject(100, 0.50001, 1.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)[0] # same stats
    False
    >>> stats_A = StatsObject(0, 0., 1.) # stats_A.N == 0
    >>> stats_B = StatsObject(50, 1., 1.)
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)[0] # no stats_A.N
    False
    >>> stats_A = StatsObject(100, 0., 1.)
    >>> stats_B = StatsObject(0, 1., 1.) # stats_B.N == 0
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)[0] # no stats_B.N
    False
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 100], [0., 0.], [1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[50, 50], [1., 0.5], [1., 5.]]])
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)
    array([ True, False], dtype=bool)
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 0, 100], [0., 0., 0.], [1., 1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[100, 100, 0], [1., 1., 1.], [1., 1., 1.]]])
    >>> minimum_significative_difference(stats_A, stats_B, alpha=0.1, beta=0.1)
    array([ True, False, False], dtype=bool)
    """
    # compute mask for zero values
    zeros_A = stats_A.N == 0
    zeros_B = stats_B.N == 0
    has_zero_N = zeros_A | zeros_B

    z_alpha = norm.ppf(1 - alpha / 2) # should be Student distribution, but degrees of freedom computation
    z_beta = norm.ppf(1 - beta)       # is a bit involved

    # compute standard errors where N != 0, np.inf where N == 0
    stderr_A = np.full(stats_A.N.shape, np.inf)
    stderr_B = np.full(stats_B.N.shape, np.inf)
    stderr_D_eq_var = np.full(stats_B.N.shape, np.inf)
    stderr_A[~zeros_A] = stats_A.std[~zeros_A] / np.sqrt(stats_A.N[~zeros_A])
    stderr_B[~zeros_B] = stats_B.std[~zeros_B] / np.sqrt(stats_B.N[~zeros_B])

    # compute combined standard error for equal and non-equal variances
    stderr_D_eq_var[~has_zero_N] = stats_A.std[~has_zero_N] * np.sqrt(1. / stats_A.N[~has_zero_N] + 1. / stats_B.N[~has_zero_N])
    stderr_D_neq_var = np.sqrt(stderr_A ** 2 + stderr_B ** 2) # not equal variance

    # compute minimal difference
    significativity_part = z_alpha * stderr_D_eq_var # minimal difference to observe for significativity
    power_part = z_beta * stderr_D_neq_var # additional difference for test power
    minimum_delta = significativity_part + power_part

    # check if current means difference is bigger than minimal significative difference
    D = np.abs(stats_A.mean - stats_B.mean)
    return minimum_delta < np.abs(D)


def two_sample_ttest(stats_A, stats_B):
    """
    Returns the p-value of the difference between two means (probability to wrongly reject the possibility that the
    difference is 0).
    Source:  https://en.wikipedia.org/wiki/Welch%27s_t_test

    >>> stats_A = StatsObject(100, 0., 1.)
    >>> stats_B = StatsObject(50, 100., 1.)
    >>> two_sample_ttest(stats_A, stats_B)[0] # doctest: +ELLIPSIS
    4.756...e-175
    >>> stats_A = StatsObject(50, 100., 1.)
    >>> stats_B = StatsObject(100, 0., 1.)
    >>> two_sample_ttest(stats_A, stats_B)[0] # doctest: +ELLIPSIS
    4.756...e-175
    >>> stats_A = StatsObject(100, 0., 1.)
    >>> stats_B = StatsObject(50, 1., 1.)
    >>> two_sample_ttest(stats_A, stats_B)[0] # doctest: +ELLIPSIS
    9.114...e-08
    >>> stats_A = StatsObject(100, 0., 1.)
    >>> stats_B = StatsObject(50, 0.5, 5.)
    >>> two_sample_ttest(stats_A, stats_B)[0] # doctest: +ELLIPSIS
    0.487...
    >>> stats_A = StatsObject(0, 0., 1.)
    >>> stats_B = StatsObject(50, 0.5, 5.)
    >>> two_sample_ttest(stats_A, stats_B)[0] # doctest: +ELLIPSIS
    1.0
    >>> stats_A = StatsObject(100, 0., 1.)
    >>> stats_B = StatsObject(0, 0.5, 5.)
    >>> two_sample_ttest(stats_A, stats_B)[0] # doctest: +ELLIPSIS
    1.0
    >>> stats_A = StatsObject(100, 1., 1.)
    >>> stats_B = StatsObject(100, 1., 1.)
    >>> two_sample_ttest(stats_A, stats_B)[0] # doctest: +ELLIPSIS
    1.0
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 100], [0., 0.], [1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[50, 50], [1., 0.5], [1., 5.]]])
    >>> two_sample_ttest(stats_A, stats_B) # doctest: +ELLIPSIS
    array([  9.114...e-08,   4.870...e-01])
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 100, 0], [0., 0., 0.], [1., 1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[50,  0,  50], [1., 1., 1.], [1., 1., 1.]]])
    >>> two_sample_ttest(stats_A, stats_B) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([  9.114...e-08,   1.0...,  1.0...])
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 0], [0., 0.], [1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[0,  50], [1., 1.], [1., 1.]]])
    >>> two_sample_ttest(stats_A, stats_B) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([ 1.,  1.])
    """
    # compute masks for zero values and extract non N=0 values
    zeros_A = stats_A.N == 0
    zeros_B = stats_B.N == 0
    has_zero_N = zeros_A | zeros_B
    stats_A.N_nnz, stats_B.N_nnz = stats_A.N[~has_zero_N], stats_B.N[~has_zero_N]
    stats_A.mean_nnz, stats_B.mean_nnz = stats_A.mean[~has_zero_N], stats_B.mean[~has_zero_N]
    stats_A.std_nnz, stats_B.std_nnz = stats_A.std[~has_zero_N], stats_B.std[~has_zero_N]

    # set default value (= 1.0) when N = 0 (total uncertainty, so probability
    # that 0 is in the confidence interval is 1.0)
    result = np.ones(stats_A.N.shape, dtype='float')

    # compute p-values where N != 0
    pvalue = ttest_ind_from_stats(stats_A.mean_nnz, stats_A.std_nnz, stats_A.N_nnz, stats_B.mean_nnz, stats_B.std_nnz, stats_B.N_nnz, equal_var=False)[1]
    result[~has_zero_N] = pvalue

    return result

def ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1):
    """
    Calculate the confidence interval for the ratio of means.
    Based on (1) https://en.wikipedia.org/wiki/Fieller%27s_theorem and
    (2) http://www.graphpad.com/FAQ/images/Ci%20of%20quotient.pdf
    To go from (1) to (2): include s in the square root, exclude a**2 from the square root,
    Q = a/b, nu_12 = 0 (samples assumed independant). sigmas from Fieller's theorem (ratio of gaussians) correspond
    to our standard errors (ratio of errors).

    >>> stats_A = StatsObject(100, 0.5, 1.)
    >>> stats_B = StatsObject(50, 1., 1.)
    >>> low, Q, high = ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    0.319...
    >>> Q[0]
    0.5
    >>> high[0] # doctest: +ELLIPSIS
    0.737...
    >>> stats_A = StatsObject(50, 1., 1.)
    >>> stats_B = StatsObject(100, 0.5, 1.)
    >>> low, Q, high = ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    1.355...
    >>> Q[0]
    2.0
    >>> high[0] # doctest: +ELLIPSIS
    3.129...
    >>> stats_A = StatsObject(100, 0.5, 1.)
    >>> stats_B = StatsObject(0, 0.5, 2.)
    >>> low, Q, high = ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    -inf
    >>> Q[0]
    nan
    >>> high[0] # doctest: +ELLIPSIS
    inf
    >>> stats_A = StatsObject(0, 0.5, 2.)
    >>> stats_B = StatsObject(100, 0.5, 1.)
    >>> low, Q, high = ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    -inf
    >>> Q[0]
    0.0
    >>> high[0] # doctest: +ELLIPSIS
    inf
    >>> stats_A = StatsObject(100, 0.5, 1.)
    >>> stats_B = StatsObject(100, 0.5, 1.)
    >>> low, Q, high = ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    0.613...
    >>> Q[0]
    1.0
    >>> high[0] # doctest: +ELLIPSIS
    1.628...
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 100], [0.5, 0.5], [1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[50, 50], [1., 0.5], [1., 2.]]])
    >>> low, Q, high = ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low # doctest: +ELLIPSIS
    array([ 0.319...,  0.460...])
    >>> Q
    array([ 0.5,  1. ])
    >>> high # doctest: +ELLIPSIS
    array([  0.737...,  14.440...])
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 100, 100], [0.5, 0.5, 1.], [1., 1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[50, 50, 50], [1., 0.1, 0.], [1., 10., 1.]]])
    >>> low, Q, high = ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([ 0.319...,  -inf, -inf])
    >>> Q
    array([ 0.5,  5. ,  nan])
    >>> high # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([  0.737...,  inf, inf])
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 0], [0.5, 0.5], [1., 1.]]]) # 2nd value: stats_A.N == 0
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[0, 50], [1., 0.1], [1., 10.]]])  # 1st value: stats_B.N == 0
    >>> low, Q, high = ratio_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([-inf, -inf])
    >>> Q
    array([ nan,   0.])
    >>> high # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([  inf,  inf])
    """
    # compute masks for zero values
    zeros_A = stats_A.N == 0
    zeros_B = stats_B.N == 0
    zeros_mean_B = stats_B.mean == 0.
    # set default values (where stats_A.N or stats_B.N or stats_B.mean = 0)
    tstar = norm.ppf(1 - alpha / 2)
    stderr_A = np.full(stats_A.N.shape, np.inf)
    stderr_B = np.full(stats_B.N.shape, np.inf)
    g = np.full(stats_B.mean.shape, np.inf)
    Q = np.ones_like(stats_A.mean)

    # stderr of N=0 sample is infinite, elsewhere it has these values:
    stderr_A[~zeros_A] = stats_A.std[~zeros_A] / np.sqrt(stats_A.N[~zeros_A])
    stderr_B[~zeros_B] = stats_B.std[~zeros_B] / np.sqrt(stats_B.N[~zeros_B])
    # Q is nan where stats_B.mean == 0 (but set to 1. temporarily to avoid propagating NaNs), elsewhere it has these values:
    Q[~zeros_mean_B] = stats_A.mean[~zeros_mean_B] / stats_B.mean[~zeros_mean_B]
    # g is infinite where stats_B.mean == 0, elsewhere it has these values:
    g[~zeros_mean_B] = (tstar * stderr_B[~zeros_mean_B] / stats_B.mean[~zeros_mean_B]) ** 2

    # all values for which g >= 1 have a zero in interval, thus CI cannot be computed for these values
    zero_in_interval = g >= 1.
    stderr_Q = np.full(Q.shape, np.inf, dtype='float')
    # we extract all values for which g < 1
    Q_nnz, g_nnz  = Q[~zero_in_interval], g[~zero_in_interval]
    stderr_A_nnz, stderr_B_nnz = stderr_A[~zero_in_interval], stderr_B[~zero_in_interval]
    stats_A.mean_nnz, stats_B.mean_nnz = stats_A.mean[~zero_in_interval], stats_B.mean[~zero_in_interval]

    # for all these values, compute the standard error of the ratio
    stderr_Q[~zero_in_interval] = (Q_nnz / (1 - g_nnz)) * np.sqrt((1 - g_nnz) *
                                    (stderr_A_nnz / stats_A.mean_nnz) ** 2 + (stderr_B_nnz / stats_B.mean_nnz) ** 2)
    # from the standard error, compute the confidence interval
    low_boundary = (Q / (1 - g)) - tstar * stderr_Q
    high_boundary = (Q / (1 - g)) + tstar * stderr_Q
    Q[zeros_A] = 0
    Q[zeros_B | zeros_mean_B] = np.nan
    return low_boundary, Q, high_boundary


def difference_of_means_confidence_interval(stats_A, stats_B, alpha=0.1):
    """
    Calculate the confidence interval for a difference of means.
    Many sources (e.g: http://www.kean.edu/~fosborne/bstat/06b2means.html)

    >>> stats_A = StatsObject(100, 0.5, 1.)
    >>> stats_B = StatsObject(50, 1., 1.)
    >>> low, Q, high = difference_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    -0.784...
    >>> Q[0]
    -0.5
    >>> high[0] # doctest: +ELLIPSIS
    -0.215...
    >>> stats_A = StatsObject(0, 0.5, 1.)
    >>> stats_B = StatsObject(50, 1., 1.)
    >>> low, Q, high = difference_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    -inf
    >>> Q[0]
    -0.5
    >>> high[0] # doctest: +ELLIPSIS
    inf
    >>> stats_A = StatsObject(100, 0.5, 1.)
    >>> stats_B = StatsObject(0, 1., 1.)
    >>> low, Q, high = difference_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    -inf
    >>> Q[0]
    -0.5
    >>> high[0] # doctest: +ELLIPSIS
    inf
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100], [0.5], [1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[50], [1.], [1.]]])
    >>> low, Q, high = difference_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    -0.784...
    >>> Q[0]
    -0.5
    >>> high[0] # doctest: +ELLIPSIS
    -0.215...
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100], [0.5], [1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[50], [0.5], [2.]]])
    >>> low, Q, high = difference_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    -0.493...
    >>> Q[0]
    0.0
    >>> high[0] # doctest: +ELLIPSIS
    0.493...
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 100], [0.5, 0.5], [1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[ 50,  50], [1.,  0.5], [1., 2.]]])
    >>> low, Q, high = difference_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low # doctest: +ELLIPSIS
    array([-0.78..., -0.49...])
    >>> Q
    array([-0.5,  0. ])
    >>> high # doctest: +ELLIPSIS
    array([-0.21...,  0.49...])
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 100, 0], [0.5, 0.5, 0.5], [1., 1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[ 50,  0, 50], [1.,  0.5, 0.5], [1., 2., 2.]]])
    >>> low, Q, high = difference_of_means_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([-0.78..., -inf, -inf])
    >>> Q
    array([-0.5,  0. ,  0. ])
    >>> high # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([-0.21...,  inf,  inf])
    """
    # compute masks for zero values
    zeros_A = stats_A.N == 0
    zeros_B = stats_B.N == 0
    # set default values (where stats_A.N or stats_B.N = 0)
    stderr_A = np.full(stats_A.N.shape, np.inf)
    stderr_B = np.full(stats_B.N.shape, np.inf)
    # compute standard error
    stderr_A[~zeros_A] = stats_A.std[~zeros_A] / np.sqrt(stats_A.N[~zeros_A])
    stderr_B[~zeros_B] = stats_B.std[~zeros_B] / np.sqrt(stats_B.N[~zeros_B])
    # compute means difference and standard error of the difference
    D = stats_A.mean - stats_B.mean
    stderr_D = np.sqrt(stderr_A ** 2 + stderr_B ** 2)
    # compute confidence interval boundaries
    tstar = norm.ppf(1 - alpha / 2)
    low_boundary = D - tstar * stderr_D
    high_boundary = D + tstar * stderr_D
    return low_boundary, D, high_boundary


def difference_of_sums_confidence_interval(stats_A, stats_B, scaling_factor=1., alpha=0.1):
    """
    Calculate the confidence interval for a difference of sums.
    Based on confidence interval formula for difference of means.
    The scaling_factor parameter is the proportionality factor to apply to the second term of the substraction (B).
    Its main usage is to estimate an increment between lopsided categories (i.e: to compute an increment for a
    80%-20% control group composition, one need to scale the control group part by scaling_factor = 80/20 = 4. to obtain
    the expected increment).

    >>> stats_A = StatsObject(100, 0.5, 1.) # 100 samples from a gaussian with mean 0 and variance 1
    >>> stats_B = StatsObject(50, 1., 1.) # 50 samples from a gaussian with mean 1 and variance 1
    >>> low, Q, high = difference_of_sums_confidence_interval(stats_A, stats_B, alpha=0.1, scaling_factor=2.)
    >>> low[0] # doctest: +ELLIPSIS
    -78.4...
    >>> Q[0]
    -50.0
    >>> high[0] # doctest: +ELLIPSIS
    -21.5...
    >>> stats_A = StatsObject(100, 0.5, 1.)
    >>> stats_B = StatsObject(0, 1., 1.)
    >>> low, Q, high = difference_of_sums_confidence_interval(stats_A, stats_B, alpha=0.1, scaling_factor=2.)
    >>> low[0] # doctest: +ELLIPSIS
    -inf
    >>> Q[0]
    50.0
    >>> high[0] # doctest: +ELLIPSIS
    inf
    >>> stats_A = StatsObject(0, 0.5, 1.)
    >>> stats_B = StatsObject(50, 1., 1.)
    >>> low, Q, high = difference_of_sums_confidence_interval(stats_A, stats_B, alpha=0.1, scaling_factor=2.)
    >>> low[0] # doctest: +ELLIPSIS
    -inf
    >>> Q[0]
    -100.0
    >>> high[0] # doctest: +ELLIPSIS
    inf
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100], [0.5], [1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[50], [1.], [1.]]])
    >>> low, Q, high = difference_of_sums_confidence_interval(stats_A, stats_B, alpha=0.1, scaling_factor=2.)
    >>> low[0] # doctest: +ELLIPSIS
    -78.4...
    >>> Q[0]
    -50.0
    >>> high[0] # doctest: +ELLIPSIS
    -21.5...
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100], [0.5], [1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[98], [0.5], [2.]]])
    >>> low, Q, high = difference_of_sums_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low[0] # doctest: +ELLIPSIS
    -35.48...
    >>> Q[0]
    1.0
    >>> high[0] # doctest: +ELLIPSIS
    37.48...
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 100], [0.5, 0.5], [1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[ 50,  50], [1.,  0.5], [1., 2.]]])
    >>> low, Q, high = difference_of_sums_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low # doctest: +ELLIPSIS
    array([-20.14...,  -3.48...])
    >>> Q
    array([  0.,  25.])
    >>> high # doctest: +ELLIPSIS
    array([ 20.14...,  53.48...])
    >>> stats_A = StatsObject(*[np.asarray(i) for i in [[100, 0, 100], [0.5, 0.5, 0.5], [1., 1., 1.]]])
    >>> stats_B = StatsObject(*[np.asarray(i) for i in [[ 50,  50, 0], [1.,  0.5, 1. ], [1., 2., 1.]]])
    >>> low, Q, high = difference_of_sums_confidence_interval(stats_A, stats_B, alpha=0.1)
    >>> low # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([-20.14...,  -inf, -inf])
    >>> Q
    array([  0., -25.,  50.])
    >>> high # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([ 20.14...,  inf, inf])
    """
    # compute sum as mean * N, then compute difference of the sums
    SA = stats_A.N * stats_A.mean
    SB = stats_B.N * stats_B.mean * scaling_factor
    diff = SA - SB

    # compute standard error of each sums and standard error of the difference
    stderr_sum_A = stats_A.std * np.sqrt(stats_A.N) # N_A * stderr(mean(A)) = N_A * ( std_A / np.sqrt(N_A) )
    stderr_sum_B = stats_B.std * np.sqrt(stats_B.N) * scaling_factor
    stderr_diff = np.sqrt(stderr_sum_A**2 + stderr_sum_B**2)

    # compute confidence interval boundaries
    tstar = norm.ppf(1 - alpha/2)
    low = diff - tstar * stderr_diff
    high = diff + tstar * stderr_diff

    # where N == 0, uncertainty is maximal so confidence interval is infinite
    zeros_mask = (stats_A.N == 0) | (stats_B.N == 0)
    low[zeros_mask] = -np.inf
    high[zeros_mask] = np.inf

    return low, diff, high
