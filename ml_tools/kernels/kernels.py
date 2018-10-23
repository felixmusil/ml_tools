
from ..base import KernelBase
from ..base import np,sp
from ..math_utils import power,average_kernel
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
        if len(X.shape) > 2:
            Nenv = X.shape[0]
            X = X.reshape((Nenv,-1))
        if Y is None:
            Y = X
        if len(Y.shape) > 2:
            Nenv = Y.shape[0]
            Y = Y.reshape((Nenv,-1))

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
            is_square = False
        else:
            Yfeat,Ystrides = None,Xstrides
            is_square = True

        N = len(Xstrides)-1
        M = len(Ystrides)-1

        envKernel = self.kernel(Xfeat,Yfeat)
        
        K = average_kernel(envKernel,Xstrides,Ystrides,is_square)

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