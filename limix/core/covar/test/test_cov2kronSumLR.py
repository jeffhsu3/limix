"""LMM testing code"""
import unittest
import scipy as sp
import numpy as np
import sys
from limix.core.covar import Cov2KronSumLR
from limix.core.covar import FreeFormCov
from limix.utils.check_grad import mcheck_grad
from limix.core.type.exception import TooExpensiveOperationError

class TestCov2KronSum(unittest.TestCase):
    def setUp(self):
        sp.random.seed(2)
        # define row caoriance
        n = 200
        f = 10
        P = 3
        X = 1.*(sp.rand(n, f)<0.2)
        # define col covariances
        Cn = FreeFormCov(P)
        self.C = Cov2KronSumLR(Cn = Cn, G = X, rank = 1)
        self.name = 'cov2kronSumLR'
        self.C.setRandomParams()

    def test_logdet_grad(self):
        def func(x, i):
            self.C.setParams(x)
            return self.C.logdet()

        def grad(x, i):
            self.C.setParams(x)
            return self.C.logdet_grad_i(i)

        x0 = self.C.getParams()
        err = mcheck_grad(func, grad, x0)
        np.testing.assert_almost_equal(err, 0., decimal = 3)

    def test_too_expensive_exceptions(self):
        n = 5001
        f = 10
        Cn = FreeFormCov(5001)
        X = 1.*(sp.rand(n, f)<0.2)
        C = Cov2KronSumLR(Cn=Cn, G=X, rank=1)
        with self.assertRaises(TooExpensiveOperationError):
            C.L()
        with self.assertRaises(TooExpensiveOperationError):
            C.R()
        with self.assertRaises(TooExpensiveOperationError):
            C.K()
        with self.assertRaises(TooExpensiveOperationError):
            C.K_grad_i(0)

    def test_param_activation(self):
        self.C.act_Cg = False
        self.C.act_Cn = False
        self.assertEqual(len(self.C.getParams()), 0)

        with self.assertRaises(ValueError):
            self.C.K_grad_i(0)

if __name__ == '__main__':
    unittest.main()
