import unittest

from labeledmatrix.core.labeledmatrix import LabeledMatrix

from tests.utils import basic_dataframe


class ConstructorsTestCase(unittest.TestCase):
    def test_from_zip_occurrence(self):
        df = basic_dataframe(100)
        lm = LabeledMatrix.from_zip_occurrence(df['a'].values, df['b'].values)
        self.assertEqual(lm.matrix.sum(),100)
        self.assertEqual(lm.row, df.drop_duplicates('a')['a'].values)
        self.assertEqual(lm.column, df.drop_duplicates('b')['b'].values)
        df1 = df.shuffle()
        lm1 = LabeledMatrix.from_zip_occurrence(df1['a'].values, df1['b'].values)
        self.assertEqual(lm1.sort(), lm.sort())
        for i in range(len(df)):
            a, b = df['a'][i], df['b'][i]
            self.assertEqual(lm1[a, b], df.counts(('a', 'b'))[(a, b)])
