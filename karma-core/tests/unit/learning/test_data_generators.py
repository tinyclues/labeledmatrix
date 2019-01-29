import unittest
import numpy as np
from karma.learning.data_generators import feedback_by_granularity_generator
from karma.core.utils.utils import use_seed
from karma.core.dataframe import DataFrame
from karma.dataframe_squash import squash
from inspect import isgeneratorfunction
from functools import partial


class FeedbackByGranularityGeneratorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with use_seed(157312):
            cls.n_samples = 1000
            cls.seed = 124325
            cls.random_df = DataFrame({'a': np.random.poisson(15, size=(cls.n_samples, 10)),
                                       'b': np.random.choice(list('abcde'), cls.n_samples),
                                       'c': np.random.poisson(15, size=(cls.n_samples, 3)),
                                       'y': np.random.randint(0, 2, cls.n_samples)})

    def test_generator(self):
        self.assertTrue(isgeneratorfunction(feedback_by_granularity_generator))

    def test_explicit(self):
        df_gen = feedback_by_granularity_generator(self.random_df, seed=self.seed, batch_shuffle=False)
        df_whole = squash(list(df_gen))
        np.testing.assert_array_equal(df_whole['c'][:], self.random_df['c'][:])
        self.assertEqual(len(df_whole), self.n_samples)

    def test_response(self):
        # Not used in explicit case, check it is not propagated
        df_gen_explicit = feedback_by_granularity_generator(self.random_df, batch_shuffle=False, seed=self.seed,
                                                            response='response')
        df_explicit = squash(list(df_gen_explicit))
        self.assertNotIn('response', df_explicit.column_names)

        # In implicit case, it is used
        df_gen_implicit = feedback_by_granularity_generator(self.random_df.copy(exclude='y'), implicit=True,
                                                            negative_positive_ratio=3, response='y_gen',
                                                            seed=self.seed, sync_blocks=(('a',), ('b',), ('c',)))
        df_implicit = squash(list(df_gen_implicit))
        self.assertIn('y_gen', df_implicit.column_names)

    def test_implicit_neg_positive_ratio(self):
        df_gen = feedback_by_granularity_generator(self.random_df.copy(exclude='y'), implicit=True, response='y_gen',
                                                   negative_positive_ratio=3, seed=self.seed,
                                                   sync_blocks=(('a',), ('b',), ('c',)))
        df = squash(list(df_gen))
        self.assertEqual(len(df.where_not('y_gen'))/len(df.where('y_gen')), 3)

        df_gen_gran = feedback_by_granularity_generator(self.random_df.copy(exclude='y'), implicit=True,
                                                        response='y_gen', negative_positive_ratio=4, seed=self.seed,
                                                        sync_blocks=(('a',), ('b',), ('c',)))
        df_gran = squash(list(df_gen_gran))
        self.assertEqual(len(df_gran.where_not('y_gen'))/len(df_gran.where('y_gen')), 4)

    def test_contrast_granularity(self):
        def _contrast_gran_comparison(df_generator, neg_pos_ratio=0):
            gran_list = []
            for df in df_generator:
                gran = np.unique(df['b'][:])
                self.assertEqual(len(gran), 1)
                gran_list.append(df['b'][:])

            count_gran_df = self.random_df.group_by('b', '# as count_gran')
            gran_list_original = map(lambda (l, n): ['{}'.format(l)] * n * (neg_pos_ratio+1),
                                     zip(count_gran_df['b'][:], count_gran_df['count_gran'][:]))
            np.testing.assert_array_equal(np.concatenate(gran_list), np.concatenate(gran_list_original))

        # explicit case
        df_gen_explicit = feedback_by_granularity_generator(self.random_df, contrast_granularity='b',
                                                            sorted_granularity=True, seed=self.seed,
                                                            batch_shuffle=True, batch_size=100)
        _contrast_gran_comparison(df_gen_explicit)

        # implicit case
        df_gen_implicit = feedback_by_granularity_generator(self.random_df.copy(exclude='y'), implicit=True,
                                                            contrast_granularity='b', seed=self.seed,
                                                            negative_positive_ratio=3,
                                                            sync_blocks=(('a',), ('b',), ('c',)),
                                                            sorted_granularity=True, batch_size=100)
        _contrast_gran_comparison(df_gen_implicit, 3)

    def test_seed_shuffle(self):
        seed = 126781
        df_gen_partial = partial(feedback_by_granularity_generator, df=self.random_df.copy(exclude='y'),
                                 implicit=True, sorted_granularity=False, batch_shuffle=True, contrast_granularity='b',
                                 sync_blocks=(('a',), ('b',), ('c',)))

        df_1, df_2, df_3 = map(lambda x: squash(list(x)), [df_gen_partial(seed=seed) for seed in 2*[seed]+[2*seed]])
        np.testing.assert_array_equal(df_1['a'][:], df_2['a'][:])
        self.assertFalse(np.all(df_1['c'][:] == df_3['c'][:]))

    def test_sync_blocks(self):
        df = self.random_df.copy()
        unique_arrays = {}
        df_gen_implicit = feedback_by_granularity_generator(self.random_df, implicit=True, seed=self.seed,
                                                            negative_positive_ratio=2, batch_shuffle=False,
                                                            sync_blocks=(('a', 'b'), ('c', 'y')),
                                                            sorted_granularity=True, batch_size=10)
        df_implicit = squash(list(df_gen_implicit))
        unique_arrays_implicit = {}
        for c1, c2 in zip('aca', 'byc'):
            df['{}_{}'.format(c1, c2)] = df['direct_sum({}, {})'.format(c1, c2)]
            unique_arrays['{}_{}'.format(c1, c2)] = len(np.unique(df['{}_{}'.format(c1, c2)][:], axis=0))

            df_implicit['{}_{}'.format(c1, c2)] = df_implicit['direct_sum({}, {})'.format(c1, c2)]
            unique_arrays_implicit['{}_{}'.format(c1, c2)] = len(np.unique(df_implicit['{}_{}'.format(c1, c2)][:],
                                                                           axis=0))

        self.assertEqual(unique_arrays_implicit['a_b'], unique_arrays['a_b'])
        self.assertEqual(unique_arrays_implicit['c_y'], unique_arrays['c_y'])
        self.assertTrue(unique_arrays_implicit['a_c'] > unique_arrays['a_c'])

    def test_batch_size(self):
        # This test has to be corrected if we want to throw away small batches in the feedback generator implementation
        df_gen_implicit_partial = partial(feedback_by_granularity_generator,
                                          **dict(df=self.random_df.copy(exclude='y'), implicit=True, seed=self.seed,
                                                 contrast_granularity='b', sorted_granularity=False, batch_shuffle=True,
                                                 sync_blocks=(('a',), ('b',), ('c',))))
        df_gen_explicit_partial = partial(feedback_by_granularity_generator,
                                          **dict(df=self.random_df.copy(exclude='y'), seed=self.seed))

        def _check_batch(df_gen_partial, batch_size, neg_pos_ratio=0):
            bs = 0
            for df in df_gen_partial(batch_size=batch_size, negative_positive_ratio=neg_pos_ratio):
                bs += len(df)
            self.assertEqual(bs, self.n_samples * (neg_pos_ratio + 1))

        with use_seed(3456):
            for _ in range(3):
                _check_batch(df_gen_implicit_partial, np.random.randint(1, 2*self.n_samples), np.random.randint(0, 10))
                _check_batch(df_gen_explicit_partial, np.random.randint(1, 2*self.n_samples))



