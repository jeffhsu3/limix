import sys
sys.path.insert(0,'./../../..')
from limix.core.type.observed import Observed
from limix.core.type.cached import *
from limix.utils.preprocess import regressOut
import scipy as sp
import numpy as np
		    
import scipy.linalg as LA
import copy
import pdb

class mean_base(cObject, Observed):

    def __init__(self,Y,F,Fstar=None):
        """
        y:        phenotype vector
        F:        fixed effect design
        """
        self.Y = Y
        self.F = F
        self.B = sp.zeros((self._K,1))
        self.Fstar = Fstar
        self.setFIinv(None)

    #########################################
    # Properties 
    #########################################
    @property
    def Y(self):
        return self._Y

    @property
    def F(self):
        return self._F

    @property
    def B(self):
        return self._B

    @property
    def B_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = sp.reshape(self.b_ste,(self._K,self._P),order='F')
        return R

    @property
    def y(self):
        return sp.reshape(self.Y,(self._N*self._P,1),order='F') 

    @property
    def b(self):
        return sp.reshape(self.B,(self._K*self._P,1),order='F')

    @property
    def b_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = sp.sqrt(self.getFIinv().diagonal())[:,sp.newaxis]
        return R

    @property
    def Fstar(self):
        return self._Fstar

    @property
    def use_to_predict(self):
        return self._use_to_predict 

    #########################################
    # Setters 
    #########################################
    @Y.setter
    def Y(self,value):
        """ set phenotype """
        self._N = value.shape[0]
        self._P = value.shape[1]
        self._Y = value
        self.clear_cache('Yres')

    @F.setter
    def F(self,value):
        """ set fixed effect design """
        assert value.shape[0]==self._N, 'Dimension mismatch'
        self._K = value.shape[1]
        self._F = value
        self.clear_cache('predict_in_sample','Yres')

    @Fstar.setter
    def Fstar(self,value):
        """ set fixed effect design for predictions """
        if value is None:
            self._use_to_predict = False
        else:
            assert value.shape[1]==self._K, 'Dimension mismatch'
            self._use_to_predict = True
        self._Fstar = value
        self.clear_cache('predict')

    @B.setter
    def B(self,value):
        """ set phenotype """
        assert value.shape[0]==self._K, 'Dimension mismatch'
        assert value.shape[1]==self._P, 'Dimension mismatch'
        self._B = value
        self.clear_cache('predict_in_sample','Yres','predict')

    @y.setter
    def y(self,value):
        """ set phenotype """
        assert value.shape[1] == 1, 'Dimension mismatch'
        self.Y = value

    @b.setter
    def b(self,value):
        """ set phenotype """
        assert value.shape[0] == self._K*self._P, 'Dimension mismatch'
        assert value.shape[1] == 1, 'Dimension mismatch'
        self.B = sp.reshape(value,(self._K,self._P),order='F')

    @use_to_predict.setter
    def use_to_predict(self,value):
        assert not (self.Fstar is None and value is True), 'set Fstar!'
        self._use_to_predict = value

    #########################################
    # Predictions 
    #########################################
    @cached
    def predict(self):
        r = _predict_fun(self.Fstar) 
        return r

    @cached
    def predict_in_sample(self):
        r = _predict_fun(self.F) 
        return r

    def _predict_fun(self,M):
        return sp.dot(M,self.B)

    @cached
    def Yres(self):
        """ residual """
        RV  = self.Y-self.predict_in_sample() 
        return RV

    #######################################
    # Standard errors
    ########################################
    def setFIinv(self, value):
        self._FIinv = value

    def getFIinv(self):
        return self._FIinv

    ###########################################
    # Gradient TODO
    ###########################################
    #def getGradient(self,j):
    #    """ get gradient for fixed effect i """
    #    return rv

    #########################################
    # Params manipulation TODO 
    #########################################
    #def getParams(self):
    #    """ get params """
    #    return rv

    #def setParams(self,params):
    #    """ set params """
    #    start = 0
    #    for i in range(self.n_terms):
    #        n_effects = self.B[i].size
    #        self.B[i] = np.reshape(params[start:start+n_effects],self.B[i].shape, order='F')
    #        start += n_effects

