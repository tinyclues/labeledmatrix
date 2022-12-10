import unittest
import numpy as np
from karma.learning.matrix_utils import coherence
from karma.core.dataframe import DataFrame
from karma.learning.benchmark_tools import descriptive_features_benchmark


class DescriptiveFeaturesBenchmarkTestCase(unittest.TestCase):
    def test_basic_columns(self):
        df = DataFrame({"col_1": np.ones((5, 2)), "col_2": np.random.randn(5, 2)})
        my_funcs = [('mean', 'l2_norm'), coherence]
        dfb_df = descriptive_features_benchmark(df, features=['col_1'], func_list=my_funcs)
        self.assertEqual(dfb_df["dimension"][:], [2])
