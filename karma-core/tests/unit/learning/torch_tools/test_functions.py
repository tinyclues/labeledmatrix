import unittest
import numpy as np
import torch
from cyperf.matrix.karma_sparse import KarmaSparse
from torch.autograd import gradcheck, Variable
from karma.learning.torch_tools.functions import SparseProjectorFunction, SparseDPDotFunction
from karma.core.utils.utils import use_seed


class SparseProjectorFunctionTestCase(unittest.TestCase):
    def test_forward(self):
        np.testing.assert_array_equal(SparseProjectorFunction.apply(KarmaSparse([[1, 4, 0, 3],
                                                                                 [2, 9, 7, 3]]),
                                                                    torch.as_tensor([3, 6, 8, 1], dtype=torch.float32)),
                                      [30, 119])

    def test_grad(self):
        with use_seed(167208):
            self.assertTrue(gradcheck(SparseProjectorFunction.apply, (KarmaSparse(np.random.rand(100, 5)),
                                                                      Variable(torch.rand(5, dtype=torch.float32),
                                                                               requires_grad=True)), eps=1e-2))


class SparseDPDotFunctionTestCase(unittest.TestCase):
    def test_forward(self):
        np.testing.assert_array_equal(SparseDPDotFunction.apply(KarmaSparse([[4, 1, 9], [0, 3, 8]]),
                                                                KarmaSparse([[9, 7], [2, 3]]),
                                                                torch.as_tensor([3, 6, 8, 1, 0, 3],
                                                                                dtype=torch.float32)), [544, 129])

    def test_grad(self):
        with use_seed(167289):
            self.assertTrue(gradcheck(SparseDPDotFunction.apply, (KarmaSparse(np.random.rand(100, 5)),
                                                                  KarmaSparse(np.random.rand(100, 3)),
                                                                  Variable(torch.rand(15, dtype=torch.float32),
                                                                           requires_grad=True)), eps=1e-2))
