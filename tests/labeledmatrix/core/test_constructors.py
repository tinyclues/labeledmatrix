import unittest

import numpy as np

from labeledmatrix.core.labeledmatrix import LabeledMatrix

from tests.utils import basic_dataframe


class ConstructorsTestCase(unittest.TestCase):
    def test_from_zip_occurrence(self):
        df = basic_dataframe(100)
        lm = LabeledMatrix.from_zip_occurrence(df['a'].values, df['b'].values)
        self.assertEqual(lm.matrix.sum(),100)
        np.testing.assert_array_equal(lm.row, df.drop_duplicates('a')['a'].values)
        np.testing.assert_array_equal(lm.column, df.drop_duplicates('b')['b'].values)
        df1 = df.sample(frac=1)
        lm1 = LabeledMatrix.from_zip_occurrence(df1['a'].values, df1['b'].values)
        self.assertEqual(lm1.sort(), lm.sort())
        df_counts = df.groupby(['a', 'b'])['a'].count().to_dict()
        for i in range(len(df)):
            a, b = df['a'][i], df['b'][i]
            self.assertEqual(lm1[a, b], df_counts[(a, b)])
