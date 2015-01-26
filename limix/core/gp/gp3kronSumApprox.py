import sys
sys.path.insert(0,'./../../..')
from gp_base import GP
from limix.core.covar import cov3kronSum
import pdb
import numpy as NP
import scipy as SP
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
import scipy.sparse as SS
import time as TIME
import copy

class gp3kronSumApprox(GP):
 
    def __init__(self,Y=None,Cr=None,Cg=None,Cn=None,XX=None,GG=None,tol=1E-3,bound='up'):
        """
        Y:      Phenotype matrix
        Cr:     LIMIX trait-to-trait covariance for region contribution
        Cg:     LIMIX trait-to-trait covariance for genetic contribution
        Cn:     LIMIX trait-to-trait covariance for noise
        XX:     Matrix for fixed sample-to-sample covariance function
        """

        # time
        self.time = SP.zeros(20)

        # dimensions
        self.N, self.P = Y.shape
    
        # pheno
        self.Y = Y
        self.y = SP.reshape(self.Y,(self.N*self.P),order='F')

        # opt
        self.tol = tol
        self.bound = bound
        
        # covars
        self.K  = cov3kronSum(Cr=Cr,Cg=Cg,Cn=Cn,GG=GG,XX=XX)

        #init cache and params
        self.cache = {} 
        self.params = None

    def setBound(self,value):
        self.bound = value
        
    def getParams(self):
        """
        get hper parameters
        """
        params = {}
        params['Cr'] = self.K.Cr.getParams()
        params['Cg'] = self.K.Cg.getParams()
        params['Cn'] = self.K.Cn.getParams()
        return params

    def setParams(self,params):
        """
        set hper parameters
        """
        self.params = params
        self.updateParams()

    def updateParams(self):
        """
        update parameters
        """
        params = SP.concatenate([self.params['Cr'],self.params['Cg'],self.params['Cn']])
        self.K.setParams(params)

    def _update_cache(self):
        """
        Update cache
        """
        cov_params_have_changed = self.K.Cr.params_have_changed or self.K.Cg.params_have_changed or self.K.Cn.params_have_changed

        if 'KiY' not in self.cache or self.XX_has_changed:
            D = SP.reshape(self.K.D(),(self.N,self.P), order='F')
            DLY = D*SP.dot(self.K.Lr(),SP.dot(self.Y,self.K.Lc().T))
            self.cache['KiY'] = SP.dot(self.K.Lr().T,SP.dot(DLY,self.K.Lc()))

        if cov_params_have_changed or self.XX_has_changed or self.GG_has_changed:
            start = TIME.time()
            self.cache['KiY'] = self.K.solve(self.Y,X0=self.cache['KiY'],tol=self.tol) 
            self.time[2]+=TIME.time()-start
        
        self.XX_has_changed = False
        self.GG_has_changed = False
        self.K.Cr.params_have_changed = False
        self.K.Cg.params_have_changed = False
        self.K.Cn.params_have_changed = False

    def LML(self,params=None,*kw_args):
        """
        calculate LML
        """
        if params is not None:
            self.setParams(params)
        self._update_cache()

        #1. constant term
        lml  = self.N*self.P*SP.log(2*SP.pi)
        #2. log det
        lml += self.K.logdet_bound(bound=self.bound)
        #3. quadratic term
        lml += (self.Y*self.cache['KiY']).sum()
        lml *= 0.5

        return lml

    def LML_debug(self,params=None,*kw_args):
        """
        calculate LML naively
        """
        assert self.N*self.P<10000, 'gp3kronSum:: N*P>=10000'

        if params is not None:
            self.setParams(params)
        self._update_cache()

        start = TIME.time()
        Rr = SP.dot(self.Xr,self.Xr.T)
        y  = SP.reshape(self.Y,(self.N*self.P), order='F') 

        K  = SP.kron(self.Cr.K(),Rr)
        K += SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K(),SP.eye(self.N))

        cholK = LA.cholesky(K)
        Kiy   = LA.cho_solve((cholK,False),y)

        lml  = y.shape[0]*SP.log(2*SP.pi)
        lml += 2*SP.log(SP.diag(cholK)).sum()
        lml += SP.dot(y,Kiy)
        lml *= 0.5
        self.time[7] = TIME.time() - start
        
        return lml

    def LMLgrad(self,params=None,**kw_args):
        """
        LML gradient
        """
        if params is not None:
            self.setParams(params)
        self._update_cache()
        RV = {}
        covars = ['Cr','Cg','Cn']
        for covar in covars:
            RV[covar] = self._LMLgrad_covar(covar)
        return RV

    def _LMLgrad_covar(self,covar,**kw_args):
        """
        calculates LMLgrad for covariance parameters
        """
        # preprocessing
        start = TIME.time()
        if covar=='Cr':
            n_params = self.K.Cr.getNumberParams()
            RKiY = SP.dot(self.K.GG,self.cache['KiY'])
        elif covar=='Cg':
            n_params = self.K.Cg.getNumberParams()
            RKiY = SP.dot(self.K.XX,self.cache['KiY'])
        elif covar=='Cn':
            n_params = self.K.Cn.getNumberParams()
            RKiY = self.cache['KiY']
        self.time[3] += TIME.time() - start

        # fill gradient vector
        RV = SP.zeros(n_params)
        for i in range(n_params):
            start = TIME.time()
            if covar=='Cr':
                C = self.K.Cr.Kgrad_param(i);
                logdetGrad = self.K.logdet_bound_grad_r(i,bound=self.bound)
            elif covar=='Cg':
                C = self.K.Cg.Kgrad_param(i);
                logdetGrad = self.K.logdet_bound_grad_g(i,bound=self.bound)
            elif covar=='Cn':
                C = self.K.Cn.Kgrad_param(i)
                logdetGrad = self.K.logdet_bound_grad_n(i,bound=self.bound)
            self.time[4] += TIME.time() - start
            
            #1. der of logdet grad 
            start = TIME.time()
            RV[i] = logdetGrad
            self.time[5]+=TIME.time()-start

            #2. der of quad term
            start = TIME.time()
            RV[i]-= SP.sum(self.cache['KiY']*SP.dot(RKiY,C.T))
            self.time[6]+=TIME.time()-start

            RV[i] *= 0.5
        
        return RV

