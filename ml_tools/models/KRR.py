from ..base import TrainerBase,RegressorBase
from ..base import np,sp
try:
    from autograd.scipy.linalg import cho_factor,cho_solve,solve_triangular
except:
    from scipy.linalg import cho_factor,cho_solve,solve_triangular

class KRR(RegressorBase):
    _pairwise = True
    
    def __init__(self,jitter,delta,trainer):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter
        self.trainer = trainer
        self.delta = delta
    
    def fit(self,kernel,y):
        '''Train the krr model with trainKernel and trainLabel.'''
        #self.X_train = X
        #kernel = self.kernel(X)
        #diag = kernel.diagonal().copy()
        #reg = np.ones(diag.shape)*np.divide(np.multiply(self.jitter  ** 2, np.mean(diag)), np.var(y))
        reg = np.ones((kernel.shape[0],))*self.jitter
        
        self.alpha = self.trainer.fit(self.delta**2*kernel,y,jitter=reg)
        
    def predict(self,kernel):
        '''kernel.shape is expected as (nPred,nTrain)'''
        #kernel = self.kernel(X,self.X_train)
        return np.dot(self.delta**2*kernel,self.alpha.flatten()).reshape((-1))
    def get_params(self,deep=True):
        return dict(sigma=self.jitter ,trainer=self.trainer,delta=self.delta)
        
    def set_params(self,params,deep=True):
        self.jitter  = params['jitter']
        self.trainer = params['trainer']
        self.delta = params['delta']
        self.alpha = None

    def pack(self):
        state = dict(weights=self.alpha,trainer=self.trainer.pack(),
                     jitter=self.jitter,delta=self.delta )
        return state
    def unpack(self,state):
        self.alpha = state['weights']
        self.delta = state['delta']
        self.trainer.unpack(state['trainer'])
        err_m = 'jitter are not consistent {} != {}'.format(self.jitter ,state['jitter'])
        assert self.jitter  == state['jitter'], err_m
    def loads(self,state):
        self.alpha = state['weights']
        self.trainer.loads(state['trainer'])
        self.jitter  = state['jitter']
        self.delta = state['delta']


class TrainerCholesky(TrainerBase):
    def __init__(self,memory_efficient):
        self.memory_eff = memory_efficient
    def fit(self,kernel,y,jitter):
        """ len(y) == len(reg)"""
        reg = jitter
        
        if self.memory_eff is True:
            kernel[np.diag_indices_from(kernel)] += reg
            kernel, lower = cho_factor(kernel, lower=False, overwrite_a=True, check_finite=False)
            L = kernel
            alpha = cho_solve((L, lower), y ,overwrite_b=False).reshape((1,-1))
        if self.memory_eff == 'autograd':
            alpha = np.linalg.solve(kernel+np.diag(reg), y).reshape((1,-1))
        else:
            L = np.linalg.cholesky(kernel+np.diag(reg))
            z = solve_triangular(L,y,lower=True)
            alpha = solve_triangular(L.T,z,lower=False,overwrite_b=True).reshape((1,-1))
       
        return alpha

    def get_params(self,deep=True):
        return dict(memory_efficient=self.memory_eff)
    
    def pack(self):
        state = dict(memory_efficient=self.memory_eff)
        return state
    def unpack(self,state):
        err_m = 'memory_eff are not consistent {} != {}'.format(self.memory_eff,state['memory_efficient'])
        assert self.memory_eff == state['memory_efficient'], err_m
    def loads(self,state):
        self.memory_eff = state['memory_efficient']


class KRRFastCV(RegressorBase):
    """ 
    taken from:
    An, S., Liu, W., & Venkatesh, S. (2007). 
    Fast cross-validation algorithms for least squares support vector machine and kernel ridge regression. 
    Pattern Recognition, 40(8), 2154-2162. https://doi.org/10.1016/j.patcog.2006.12.015
    """
    _pairwise = True
    
    def __init__(self,jitter,delta,cv):
        self.jitter = jitter
        self.cv = cv
        self.delta = delta
    
    def fit(self,kernel,y):
        '''Fast cv scheme. Destroy kernel.'''
	invkernel = kernel.copy()
        np.multiply(self.delta**2,invkernel,out=invkernel)
        invkernel[np.diag_indices_from(invkernel)] += self.jitter
        invkernel = np.linalg.inv(invkernel)
        alpha = np.dot(invkernel,y)
        Cii = []
        beta = np.zeros(alpha.shape)
        self.y_pred = np.zeros(y.shape)
        self.error = np.zeros(y.shape)
        for _,test in self.cv.split(invkernel):
            Cii = invkernel[np.ix_(test,test)]
            beta = np.linalg.solve(Cii,alpha[test]) 
            self.y_pred[test] = y[test] - beta
            self.error[test] = beta # beta = y_true - y_pred 

        del invkernel
        
    def predict(self,kernel=None):
        '''kernel.shape is expected as (nPred,nTrain)'''
        #kernel = self.kernel(X,self.X_train)
        return self.y_pred

    def get_params(self,deep=True):
        return dict(sigma=self.jitter ,cv=self.cv)
        
    def set_params(self,params,deep=True):
        self.jitter  = params['jitter']
        self.cv = params['cv']
        self.delta = params['delta']
        self.y_pred = None

    def pack(self):
        state = dict(y_pred=self.y_pred,cv=self.cv.pack(),
                     jitter=self.jitter,delta=self.delta )
        return state
    def unpack(self,state):
        self.y_pred = state['y_pred']
        self.cv.unpack(state['cv'])
        self.delta = state['delta']

        err_m = 'jitter are not consistent {} != {}'.format(self.jitter ,state['jitter'])
        assert self.jitter  == state['jitter'], err_m
    def loads(self,state):
        self.y_pred = state['y_pred']
        self.cv.loads(state['cv'])
        self.jitter  = state['jitter']
        self.delta = state['delta']
