import sys
sys.path.append('./..')
sys.path.append('./../../..')

import scipy as SP
import scipy.linalg
import limix
import pdb
import time



class CVarianceDecomposition:
    """
    helper function for variance decomposition in limix
    This class mainly takes care of initialization and interpreation of results
    """
    
    """
    Methods:
        __init__(self,Y,T,F,C,F):    Constructor
        initialise:                  Initialise parameters
        fit:                         Fit phenos and returns the minimum with all the info
        fit_ntimes:                  Fit phenos ntimes with different random initialization and returns the minima order with respect to gp.LML
        getGP:                       Return the GP of the limix variacne decomposition class
        getParams:                   Returns the stack vector of the parameters of the trait covariances C_i
        getEmpTraitCov:              Returns the empirical trait covariance
        getEstTraitCov:              Returns the total trait covariance in the GP by summing up the trait covariances C_i
        getCovParams:                Calculates the inverse hessian of the -loglikelihood with respect to the parameters: this constitutes the covariance matrix of the posterior over paramters under laplace approximation
        estimateHeritabilities:      It fits the model with 1 global fixed effects and covariance matrix h1[p]*K+h2[p] for each trait p and returns the vectors h1 and h2
        exportMin:                   Export information about the min in the given h5py file or group
        exportMins:                  Export information about all the min in the given h5py file or group

    """
    
    def __init__(self,Y,T,F,C,K):
        """
        Y: phenotype matrix
        T: trait matrix / list of trait matrices
        F: fixed effect
        C: list of trait covariances
        K: list of kronecker matrices
        """
        
        # Number of terms
        self.n_terms = len(C)
        
        if type(T)!=list:
            Tl = []
            for i in range(self.n_terms):
                Tl.append(T)
            T = Tl
        
        #create column of 1 for fixed if nothing providede
        self.N       = (T[0]==0).sum()
        self.P       = int(T[0].max()+1)
        self.Nt      = self.N*self.P
        
        assert len(K)==len(C), 'K and C must have the same length'
        
        assert Y.shape[0]==self.Nt, 'outch'
        assert Y.shape[1]==1, 'outch'
        
        for i in range(self.n_terms):
            assert K[i].shape[0]==self.Nt, 'outch'
            assert K[i].shape[1]==self.Nt, 'outch'
            assert T[i].shape[1]==1, 'outch'

        if F==None:
            F=SP.ones(self.Nt)
                
        #trait and phenotype
        self.Y = Y
        self.T = T
        self.F = F
        self.K = K
        self.C = C
        
        pass
    
    
    def initialise(self):
        """
        get random initialization of variances based on the empirical trait variance
        """
        EmpVarY=self.getEmpTraitVar()
        temp=SP.rand(self.n_terms,self.P)
        N=temp.sum(0)
        temp=temp/N*EmpVarY
        for term_i in range(self.n_terms):
            if self.C[term_i].getName()=='CTDenseCF':
                params=(2*SP.rand(self.P)-1)*SP.sqrt(temp[term_i,:])
                self.C[term_i].setParams(params)
            elif self.C[term_i].getName()=='CTFixedCF':
                params=SP.array([SP.sqrt(temp[term_i,:].mean())])
                self.C[term_i].setParams(params)
            elif self.C[term_i].getName()=='CTDiagonalCF':
                params=SP.sqrt(temp[term_i,:])
                self.C[term_i].setParams(params)
            else:
                print 'Not implemented for %s' % self.C[term_i].getName()
                break

    
    def fit(self,grad_threshold=1e-2):
        """
        fit a variance component model with the predefined design and the initialization and returns all the results
        """
        
        # Storing some meaning full information
        Params0 = self.getParams()
    
        # LIMIX CVARIANCEDECOMPOSITION INITIALIZATION
        vt = limix.CVarianceDecomposition()
        vt.setPheno(self.Y)
        vt.setFixed(self.F)
        for term_i in range(self.n_terms):
            vt.addCVTerm(self.C[term_i],self.K[term_i],self.T[term_i])
        vt.initGP()
    
        # GET GP AND STORE LML0
        gp=vt.getGP()
        LML0=-1.0*gp.LML()
    
        # LIMIX CVARIANCEDECOMPOSITION FITTING
        start_time = time.time()
        conv=vt.train()
        time_train=time.time()-start_time
    
        # Takes the estimated Trait Covariance Matrix
        TraitCovar=self.getEmpTraitCov()
        
        # Check whether limix::CVarianceDecomposition.train() has converged
        ParamMask=gp.getParamMask()['covar']
        LMLgrad = SP.linalg.norm(gp.LMLgrad()['covar'][ParamMask==1])
        if conv!=True or LMLgrad>grad_threshold or self.getParams().max()>10*SP.sqrt(self.getEmpTraitVar().max()):
            print 'limix::CVarianceDecomposition::train has not converged'
            res=None
        else:
            res = {
                'Params0':          Params0,
                'Params':           self.getParams(),
                'LML':              SP.array([-1.0*gp.LML()]),
                'LML0':             SP.array([LML0]),
                'LMLgrad':          SP.array([LMLgrad]),
                'time_train':       SP.array([time_train]),
                'TraitCovar':       TraitCovar,
                'gp' :              gp
                }
        return res
        pass
    
    
    def fit_ntimes(self,ntimes=10,grad_threshold=1e-2,dist_mins=1e-2):
        """
        fit phenos ntimes with different random initialization and returns the minima order with respect to gp.LML
        """

        optima=[]
        LML=SP.zeros((1,0))
        
        for i in range(ntimes):
            
            print ".. Minimization %d" % i
            
            self.initialise()
            min=self.fit(grad_threshold=grad_threshold)
        
            if min!=None:
                temp=1
                for j in range(len(optima)):
                    if SP.linalg.norm(min['Params']-optima[j]['Params'])<dist_mins:
                        temp=0
                        optima[j]['counter']+=1
                        break
                if temp==1:
                    min['counter']=1
                    optima.append(min)
                    LML=SP.concatenate((LML,min['LML'][:,SP.newaxis]),1)
    
         # Order the list optima with respect to LML the first optimum has highest LML
        optima1=[]
        index = LML.argsort(axis = 1)[0,:][::-1]
        for i in range(len(optima)):
            optima1.append(optima[index[i]])
    
        return optima1

    
    def getGP(self):
        """
        Returns the GP of the limix class CVarianceDecomposition
        """
        vt = limix.CVarianceDecomposition()
        vt.setPheno(self.Y)
        vt.setFixed(self.F)
        for term_i in range(self.n_terms):
            vt.addCVTerm(self.C[term_i],self.K[term_i],self.T[term_i])
        vt.initGP()
        gp=vt.getGP()
        return gp

    def getParams(self):
        """
        Returns the Parameters
        """
        params=SP.concatenate([self.C[term_i].agetScales() for term_i in range(self.n_terms)])
        return params

    def getEstTraitCov(self,terms=None):
        """
        Returns the estimated trait covariance matrix
        terms: index of temrs to use for this estimate
        """
        if terms is None:
            terms = xrange(self.n_terms)
        TraitCovar=SP.zeros((self.P,self.P))
        for term_i in terms:
            TraitCovar+=self.C[term_i].getK0()
        return TraitCovar

    def getCovParams(self,min):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE COVARIANCE MATRIX OF THE OPTIMIZED PARAMETERS
        """
        gp=min['gp']
        ParamMask=gp.getParamMask()['covar']
        std=SP.zeros(ParamMask.sum())
        H=gp.LMLhess(["covar"])
        It= (ParamMask[:,0]==1)
        H=H[It,:][:,It]
        Sigma = SP.linalg.inv(H)
        return Sigma
    
    def getModelPosterior(self,min,Sigma=None):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE BAYESIAN MODEL POSTERIOR
        """
        
        if Sigma==None:
            Sigma = self.getCovParams(min)
        
        n_params = 0
        for term_i in range(self.n_terms):
            n_params += self.C[term_i].getNumberParams()
        
        ModCompl = 0.5*n_params*SP.log(2*SP.pi)+0.5*SP.log(SP.linalg.det(Sigma))
        
        RV = min['LML']+ModCompl
            
        return RV
    

    def getEmpTraitCov(self):
        """
        Returns the empirical trait covariance matrix
        """
        Y1=(self.Y).reshape((self.P,self.N))
        RV=SP.cov(Y1).reshape([self.P,self.P])
        return RV
    
    def getEmpTraitVar(self):
        """
        Returns the vector of empirical trait variances
        """
        if self.P==1:
            RV = self.getEmpTraitCov()
        else:
            RV = self.getEmpTraitCov().diagonal()
        return RV




    def estimateHeritabilities(self,Kpop):
        """
        It fits the model with 1 fixed effects and covariance matrix h1[p]*K+h2[p] for each trait p and return the vectors h1 and h2
        """
    
        h1 = SP.zeros(self.P)
        h2 = SP.zeros(self.P)
    
        for p in range(self.P):
            It = (self.T[0][:,0]==p)
            K= Kpop[It,:][:,It]
            N= K.shape[0]
            y=self.Y[It]
            h1[p], h2[p]=limix.CVarianceDecomposition.aestimateHeritability(y,SP.ones(N),K)
            if h1[p]<1e-6:   h1[p]=1e-6
            if h2[p]<1e-6:   h2[p]=1e-6
    
        return h1, h2
    
    def exportMin(self,min,f,counter=0,laplace=0):
        """
        Export the min in the given h5py file or group    
        """
        f.create_dataset("Params0",data=min["Params0"])
        f.create_dataset("LML0",data=min["LML0"])
        f.create_dataset("Params",data=min["Params"])
        f.create_dataset("LML",data=min["LML"])
        f.create_dataset("LMLgrad",data=min["LMLgrad"])
        f.create_dataset("time_train",data=min["time_train"])
        if counter!=0:
            f.create_dataset("counter",data=min["counter"])
        if laplace!=0:
            covParams = self.getCovParams(min)
            f.create_dataset("covParams",data=covParams)
            f.create_dataset("modelPosterior",data=self.getModelPosterior(min,Sigma=covParams))
        
    
    def exportMins(self,mins,f,laplace=0):
        """
        Export all the min in the given h5py file or group  
        """
        for min_i in range(len(mins)):
            g=f.create_group('min%d'%min_i)
            self.exportMin(mins[min_i],g,counter=1,laplace=laplace)



