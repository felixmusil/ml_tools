
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


class KernelSum(KernelBase):
    def __init__(self,kernel):
        self.kernel = kernel
        
    def fit(self,X):   
        return self
    def get_params(self,deep=True):
        params = dict(kernel=self.kernel,strides=self.strides)
        return params
    def set_params(self,**params):
        self.kernel = params['kernel']
        self.delta = self.kernel.delta
        self.strides = params['strides']
        
    def transform(self,X,X_train=None):
        Xfeat,Xstrides = X['feature_matrix'], X['strides']
        
        if X_train is not None:
            Yfeat,Ystrides = X_train['feature_matrix'], X_train['strides']
        else:
            Yfeat,Ystrides = None,Xstrides

        N = len(Xstrides)-1
        M = len(Ystrides)-1

        envKernel = self.kernel(Xfeat,Yfeat)
        
        K = np.zeros((N,M))
        for ii,(ist,ind) in enumerate(zip(Xstrides[:-1],Xstrides[1:])):
            for jj,(jst,jnd) in enumerate(zip(Ystrides[:-1],Ystrides[1:])):
                K[ii,jj] = np.mean(envKernel[ist:ind,jst:jnd])
        return K

    def pack(self):
        state = self.get_params()
        return state
    def unpack(self,state):
        pass
    
    def loads(self,state):
        self.set_params(state)

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
        
    def transform(self,X,y=None,X_train=None,X_pseudo=None):
        if X_train is None and isinstance(X,dict) is False and y is not None:
            if X_pseudo is None:
                Xs = X[self.pseudo_input_ids]
            else:
                Xs = X_pseudo

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