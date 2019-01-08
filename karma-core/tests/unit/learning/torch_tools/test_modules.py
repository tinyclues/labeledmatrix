import unittest
import numpy as np
import torch

from functools import partial
from cyperf.matrix.karma_sparse import KarmaSparse

from karma.core.dataframe import DataFrame
from karma.core.utils.utils import use_seed
from karma.learning.torch_tools.modules import KarmaLinearModule, LinearCell, DeepFM, DeepLayersCell,\
    ConcatBiInteractionCell, CELL_AVAILABLE_INTERACTIONS

with use_seed(25738):
    random_df = DataFrame({'a': KarmaSparse(np.random.poisson(15, size=(1000, 10))),
                           'b': KarmaSparse(np.random.poisson(15, size=(1000, 3)).astype(np.float32)),
                           'c': KarmaSparse(np.random.poisson(15, size=(1000, 14))),
                           'd': np.random.poisson(15, size=(1000, 20)).astype(np.float32),
                           'y': np.random.randint(0, 2, 1000)})


class KarmaLinearModuleTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = random_df.copy()
        cls.karma_lin_module_0 = KarmaLinearModule(10, out_dim=2, bias=True)
        cls.karma_lin_module_1 = KarmaLinearModule(14, out_dim=None, bias=False)

    def test_parameters(self):
        np.testing.assert_array_equal(self.karma_lin_module_0(self.df['a'][:]).size(), [len(self.df), 2])
        self.assertEqual(sorted(self.karma_lin_module_0.params_dict.keys()), ['bias', 'projector'])
        np.testing.assert_array_equal(self.karma_lin_module_0.bias.size(), 2)
        np.testing.assert_array_equal(self.karma_lin_module_0.projector.size(), [10, 2])

        np.testing.assert_array_equal(self.karma_lin_module_1(self.df['c'][:]).size(), len(self.df))
        self.assertEqual(sorted(self.karma_lin_module_1.params_dict.keys()), ['projector'])
        np.testing.assert_array_equal(self.karma_lin_module_1.projector.size(), 14)

    def test_kc_forward(self):
        self.df += (self.karma_lin_module_0.to_karmacode('a') + self.karma_lin_module_1.to_karmacode('c'))
        for module, inp in zip([self.karma_lin_module_0, self.karma_lin_module_1], 'ac'):
            np.testing.assert_array_equal(module(self.df[inp][:]).detach(), self.df['linear_{}'.format(inp)][:])


class LinearCellTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = random_df.copy()
        cls.size_dict = {col: random_df[col].safe_dim() for col in ['a', 'c']}
        cls.linear_cell_0 = LinearCell(cls.size_dict, bias=True)
        cls.linear_cell_1 = LinearCell(cls.size_dict, bias=False)

    def test_parameters(self):
        np.testing.assert_array_equal(self.linear_cell_0(self.df).size(), len(self.df))
        np.testing.assert_array_equal(np.sort(np.reshape([p.size()
                                                          for p in self.linear_cell_0.params_dict.values()], -1)),
                                      [1, 10, 14])

    def test_kc_forward(self):
        for i in range(2):
            linear_cell = getattr(self, 'linear_cell_{}'.format(i))
            self.df += linear_cell.to_karmacode(out='out_{}'.format(i))
            np.testing.assert_array_equal(linear_cell(self.df).detach(), self.df['out_{}'.format(i)][:].reshape(-1))


class DeepLayersCellTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = random_df.copy()
        cls.deep_0 = DeepLayersCell(sizes=[20])
        cls.deep_1 = DeepLayersCell(sizes=[20, 10], activation='tanh')

    def test_parameters(self):
        self.assertEqual(len(self.deep_0.params_dict.keys()), 2)
        np.testing.assert_array_equal(self.deep_0(torch.as_tensor(self.df['d'][:])).size(), [len(self.df), 1])
        np.testing.assert_array_equal(self.deep_0.deep_layer_0.weight.size(), [1, 20])

        self.assertEqual(len(self.deep_1.params_dict.keys()), 3)
        np.testing.assert_array_equal(self.deep_1(torch.as_tensor(self.df['d'][:])).size(), [len(self.df), 1])
        np.testing.assert_array_equal(self.deep_1.deep_layer_0.weight.size(), [10, 20])
        np.testing.assert_array_equal(self.deep_1.deep_layer_1.weight.size(), [1, 10])

    def test_kc_forward(self):
        for i in range(2):
            deep = getattr(self, 'deep_{}'.format(i))
            self.df += deep.to_karmacode(inp='d', out='deep_{}'.format(i))
            np.testing.assert_array_equal(deep(torch.as_tensor(self.df['d'][:])).detach(),
                                          self.df['deep_{}'.format(i)][:])


class ConcatBiInteractionCellTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = random_df.copy()
        cls.embed_dim = 4
        cls.config = ('a', ('b', 'c'))
        cls.size_dict = {col: random_df[col].safe_dim() for col in random_df.sparse_column_names}
        cls.bi_inter_partial = partial(ConcatBiInteractionCell, config=cls.config, size_dict=cls.size_dict,
                                       embed_dim=cls.embed_dim)

    def test_parameters(self):
        for interaction, out_dim in zip(CELL_AVAILABLE_INTERACTIONS, [self.embed_dim, self.embed_dim**2, 1]):
            bi_inter = self.bi_inter_partial(interaction=interaction, identifier=0)
            self.assertEqual(len(bi_inter.params_dict.keys()), 3)
            np.testing.assert_array_equal(bi_inter(self.df).size(), [len(self.df), out_dim])
            np.testing.assert_array_equal(bi_inter.PRLeft_a_0.size(), [10, self.embed_dim])
            np.testing.assert_array_equal(bi_inter.PRRight_b_0.size(), [3, self.embed_dim])

    def test_kc_forward(self):
        for interaction, out_dim in zip(CELL_AVAILABLE_INTERACTIONS, [self.embed_dim, self.embed_dim ** 2, 1]):
            bi_inter = self.bi_inter_partial(interaction=interaction)
            self.df += bi_inter.to_karmacode(out='out_{}'.format(interaction))
            np.testing.assert_array_almost_equal(bi_inter(self.df).detach(),
                                                 self.df['out_{}_{}'.format(interaction,
                                                                            bi_inter.id)][:].reshape(-1, out_dim),
                                                 decimal=3)

    def test_identifier(self):
        for id in range(2):
            bi_inter = self.bi_inter_partial(identifier=id)
            for param in bi_inter.params_dict.keys():
                self.assertEqual(int(param[-1]), id)
            self.assertEqual(int(bi_inter.to_karmacode().outputs[0][-1]), id)
            self.assertEqual(int(bi_inter.to_karmacode(out='some_output').outputs[0][-1]), id)


class DeepFMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = random_df.copy()
        cls.model_kwargs = {'deep_layer': {'activation': 'relu', 'hidden_sizes': [15, 12]},
                            'fit_linear': True,
                            'size_dict': {col: random_df[col].safe_dim() for col in cls.df.sparse_column_names},
                            'seed': 4321,
                            'features': random_df.sparse_column_names
                            }
        config_1 = ('a', ('b', 'c'))
        config_2 = ('b', 'c')
        cls.model_kwargs['base_cells'] = [{'type': 'concat_bi_interaction_cell',
                                           'cell_kwargs': dict(config=config,
                                                               embed_dim=embed_dim,
                                                               interaction='multiply')}
                                          for config, embed_dim in zip([config_1, config_2], [2, 3])]

    def test_parameters(self):
        deep_fm = DeepFM(self.model_kwargs)
        # Not adding any parameters compared to children modules, therefore everything has been checked before
        self.assertEqual(len(deep_fm.params_dict.keys()),
                         sum([len(c.params_dict.keys()) for c in deep_fm.children_dict.values()]))

    def test_kc_forward(self):
        deep_fm = DeepFM(self.model_kwargs)
        self.df += deep_fm.to_karmacode(out='out_deepfm')
        np.testing.assert_array_almost_equal(deep_fm(self.df).detach(), self.df['out_deepfm'][:].reshape(-1),
                                             decimal=3)
