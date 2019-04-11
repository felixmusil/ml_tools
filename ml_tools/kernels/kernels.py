
from ..base import KernelBase
from ..base import np,sp
from ..math_utils import power
from scipy.sparse import issparse


class KernelPower(KernelBase):
    def __init__(self,zeta):
        self.zeta = zeta
    def fit(self,X):   
        return self
    def get_params(self,deep=True):
        params = dict(zeta=self.zeta)
        return params
    def set_params(self,**params):
        self.zeta = params['zeta']
    def transform(self,X,X_train=None):
        if X_train is None:
            return self(X)
        else: 
            return self(X,Y=X_train)
        
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        if issparse(X) is False:
            return power(np.dot(X,Y.T),self.zeta)
        if issparse(X) is True:
            N, M = X.shape[0],Y.shape[0]
            kk = np.zeros((N,M))
            X.dot(Y.T).todense(out=kk)
            return power(kk,self.zeta)
          
    def pack(self):
        state = dict(zeta=self.zeta)
        return state
    def unpack(self,state):
        err_m = 'zetas are not consistent {} != {}'.format(self.zeta,state['zeta'])
        assert self.zeta == state['zeta'], err_m
        
    def loads(self,state):
        self.zeta = state['zeta']


class KernelPP(KernelBase):
    """
    Class allowing projected process schemes to perform a kernel ridge regression.
    It uses the KernelZeta infrastructure and needs in input a list of configuration the user
    will use as active set, it has to be a selection of the input elements, in the form a list of integers
    Andrea Anelli 2019 18/03
    """
    def __init__(self,zeta,active_idx):
        self.zeta = zeta
        self.active_idx = active_idx
    def fit(self,X):   
        return self
    def get_params(self,deep=True):
        params = dict(zeta=self.zeta,active_idx=self.active_idx)
        return params
    def set_params(self,**params):
        self.zeta = params['zeta']
        self.active_idx = params['active_idx']
    def transform(self,X,X_train=None):
        if X_train is None:
            return self(X)
        else: 
            return self(X,Y=X_train)
        
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
            Kmm = power(np.dot(X[self.active_idx],Y[self.active_idx].T),self.zeta)
            Kmn = power(np.dot(X[self.active_idx],Y.T),self.zeta)
            return Kmm,Kmn
        else: 
            Kmx = power(np.dot(X,Y[self.active_idx].T),self.zeta)
            return Kmx
          
    def pack(self):
        state = dict(zeta=self.zeta,active=self.active_idx)
        return state
    def unpack(self,state):
        err_m = 'zetas are not consistent {} != {}'.format(self.zeta,state['zeta'])
        assert self.zeta == state['zeta'], err_m
        
    def loads(self,state):
        self.zeta = state['zeta']
        self.active = state['active_idx']


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
    def __init__(self,kernel,X_pseudo,Lambda):
        self.Lambda = Lambda # \sim std of the training properties
        self.kernel = kernel
        self.X_pseudo = X_pseudo
        
    def fit(self,X):   
        return self
    def get_params(self,deep=True):
        params = dict(kernel=self.kernel,X_pseudo=self.X_pseudo)
        return params
    def set_params(self,**params):
        self.kernel = params['kernel']
        self.X_pseudo = params['X_pseudo']
        
    def transform(self,X,y=None,X_train=None):
        if X_train is None and isinstance(X,dict) is False and y is not None:
            Xs = self.X_pseudo
            
            kMM = self.kernel(Xs,Y=Xs)
            kMN = self.kernel(Xs,Y=X)
            ## assumes Lambda= Lambda**2*np.diag(np.ones(n))
            sparseK = kMM + np.dot(kMN,kMN.T)/self.Lambda**2
            sparseY = np.dot(kMN,y)/self.Lambda**2

            return sparseK,sparseY

        else: 
            return self.kernel(X,Y=self.X_pseudo)

    def pack(self):
        state = self.get_params()
        return state
    def unpack(self,state):
        pass
    
    def loads(self,state):
        self.set_params(state)
