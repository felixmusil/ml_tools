
from ..base import KernelBase
from ..base import np,sp
from ..math_utils import power,average_kernel
#from ..math_utils.basic import power,average_kernel
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
        # X should be shape=(Nsample,Mfeature)
        # if not assumes additional dims are features
        if len(X.shape) > 2:
            Nenv = X.shape[0]
            Xi = X.reshape((Nenv,-1))
        else:
            Xi = X
        if Y is None:
            Yi = Xi
        elif len(Y.shape) > 2:
            Nenv = Y.shape[0]
            Yi = Y.reshape((Nenv,-1))
        else:
            Yi = Y

        if issparse(Xi) is False:
            return power(np.dot(Xi,Yi.T),self.zeta)
        if issparse(Xi) is True:
            N, M = Xi.shape[0],Yi.shape[0]
            kk = np.zeros((N,M))
            Xi.dot(Yi.T).todense(out=kk)
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
        params = dict(kernel=self.kernel)
        return params
    def set_params(self,**params):
        self.kernel = params['kernel']
        
    def transform(self,X,X_train=None):
        Xfeat,Xstrides = X['feature_matrix'], X['strides']
        
        if X_train is not None:
            Yfeat,Ystrides = X_train['feature_matrix'], X_train['strides']
            is_square = False
        else:
            Yfeat,Ystrides = None,Xstrides
            is_square = True
 
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
        params = dict(kernel=self.kernel,X_pseudo=self.X_pseudo,
                    Lambda=self.Lambda)
        return params
    def set_params(self,**params):
        self.kernel = params['kernel']
        self.X_pseudo = params['X_pseudo']
        self.Lambda = params['Lambda']
        
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