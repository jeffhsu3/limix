"""Variance Decomposition testing code"""
import unittest
import scipy as SP
import scipy.stats
import pdb
import os
import sys
import limix

geno_file = './varDecomp/genotype.csv'
pheno_file = './varDecomp/phenotype.csv'
param_file = './varDecomp/params_true.csv'

class CVarianceDecomposition_test(unittest.TestCase):
    """test class for CVarianceDecomposition"""

    def genGeno(self):
        self.X  = (SP.rand(self.N,self.S)<0.2)*1.

    
    def genPheno(self):
        dp = SP.ones(self.P); dp[1]=-1
        Y = SP.zeros((self.N,self.P))
        gamma0_g = SP.randn(self.S)
        gamma0_n = SP.randn(self.N)
        for p in range(self.P):
            gamma_g = SP.randn(self.S)
            gamma_n = SP.randn(self.N)
            beta_g  = dp[p]*gamma0_g+gamma_g
            beta_n  = gamma0_n+gamma_n
            y=SP.dot(self.X,beta_g)
            y+=beta_n
            Y[:,p]=y
        self.Y=SP.stats.zscore(Y,0)
    
    def setUp(self):
        #check: do we have a csv File?
        if ((not os.path.exists(geno_file)) | (not os.path.exists(pheno_file))) or 'recalc' in sys.argv:
            SP.random.seed(1)
            self.N = 200
            self.S = 1000
            self.P = 2
            self.genGeno()
            self.genPheno()
            self.generate = True
            SP.savetxt(geno_file,self.X)
            SP.savetxt(pheno_file,self.Y)
        else:
            self.generate=False
            self.X = SP.loadtxt(geno_file)
            self.Y = SP.loadtxt(pheno_file)
            self.N = self.X.shape[0]
            self.S = self.X.shape[1]
            self.P = self.Y.shape[1]

        self.Kg = SP.dot(self.X,self.X.T)

        self.vd = limix.CVarianceDecomposition(self.Y)
        self.vd.addFixedEffTerm("specific")
        self.vd.addTerm("Dense",self.Kg)
        self.vd.addTerm("Block",self.Kg)
        self.vd.addTerm("Dense",SP.eye(self.N))
        self.vd.addTerm("Identity",SP.eye(self.N))
        self.vd.initGP()
        
    def test_fit(self):
        """ optimization test """
        self.vd.trainGP()
        params = self.vd.getOptimum()['scales'][:,0]
        if self.generate:
            SP.savetxt(param_file,params)
            params_true = SP.zeros_like(params)
        else:
            params_true = SP.loadtxt(param_file)
        RV = (params-params_true).max()<1e-6
        self.assertTrue(RV)


if __name__ == '__main__':
    unittest.main()

