
from ..base import KernelBase
import numpy as np
try:
    from ..math_utils.optimized import power
except:
    from ..math_utils.basic import power



class KernelPower(KernelBase):
    def __init__(self,zeta,delta):
        self.zeta = zeta
        self.delta = delta
    def fit(self,X):   
        return self
    def get_params(self,deep=True):
        params = dict(zeta=self.zeta,delta=self.delta)
        return params
    def set_params(self,**params):
        self.zeta = params['zeta']
        self.delta = params['delta']
    def transform(self,X,X_train=None):
        if X_train is None:
            return self(X)
        else: 
            return self(X,Y=X_train)
        
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        return np.multiply(power(np.dot(X,Y.T),self.zeta),self.delta**2)
          
    def pack(self):
        state = dict(zeta=self.zeta,delta=self.delta)
        return state
    def unpack(self,state):
        err_m = 'zetas are not consistent {} != {}'.format(self.zeta,state['zeta'])
        assert self.zeta == state['zeta'], err_m
        err_m = 'delta are not consistent {} != {}'.format(self.delta,state['delta'])
        assert self.delta == state['delta'], err_m
    def loads(self,state):
        self.zeta = state['zeta']
        self.delta = state['delta']



class KernelSparseSoR(KernelBase):
    def __init__(self,kernel,pseudo_input_ids=None):
        self.delta = kernel.delta # \sim std of the training properties
        self.kernel = kernel
        self.pseudo_input_ids = pseudo_input_ids
        
    def fit(self,X):   
        return self
    def get_params(self,deep=True):
        params = dict(kernel=self.kernel,pseudo_input_ids=self.pseudo_input_ids)
        return params
    def set_params(self,**params):
        self.kernel = params['kernel']
        self.delta = self.kernel.delta
        self.pseudo_input_ids = params['pseudo_input_ids']
        
    def transform(self,X,y=None,X_train=None):
        if X_train is None and isinstance(X,dict) is False and y is not None:
            Xs = X[self.pseudo_input_ids]
            kMM = self.kernel(Xs,Y=Xs)
            kMN = self.kernel(Xs,Y=X)
            ## assumes Lambda= delta**2*np.diag(np.ones(n))
            sparseK = kMM + np.dot(kMN,kMN.T)/self.delta**2
            sparseY = np.dot(kMN,y)/self.delta**2

            return sparseK,sparseY
        else: 
            return self.kernel(X,Y=X_train)


    def pack(self):
        state = self.get_params()
        return state
    def unpack(self,state):
        pass
    
    def loads(self,state):
        self.set_params(state)