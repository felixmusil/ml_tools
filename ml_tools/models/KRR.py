import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular,cho_factor
from ..base import TrainerBase,RegressorBase


class KRR(RegressorBase):
    _pairwise = True
    
    def __init__(self,jitter,trainer):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter
        self.trainer = trainer
    
    def fit(self,kernel,y):
        '''Train the krr model with trainKernel and trainLabel.'''
        #self.X_train = X
        #kernel = self.kernel(X)
        #diag = kernel.diagonal().copy()
        #reg = np.ones(diag.shape)*np.divide(np.multiply(self.jitter  ** 2, np.mean(diag)), np.var(y))
        reg = np.ones((kernel.shape[0],))*self.jitter
        
        self.alpha = self.trainer.fit(kernel,y,jitter=reg)
        
    def predict(self,kernel):
        '''kernel.shape is expected as (nPred,nTrain)'''
        #kernel = self.kernel(X,self.X_train)
        return np.dot(kernel,self.alpha.flatten()).reshape((-1))
    def get_params(self,deep=True):
        return dict(sigma=self.jitter ,trainer=self.trainer)
        
    def set_params(self,params,deep=True):
        self.jitter  = params['jitter']
        self.trainer = params['trainer']
        self.alpha = None

    def pack(self):
        state = dict(weights=self.alpha,trainer=self.trainer.pack(),
                     jitter=self.jitter )
        return state
    def unpack(self,state):
        self.alpha = state['weights']
        self.trainer.unpack(state['trainer'])
        err_m = 'jitter are not consistent {} != {}'.format(self.jitter ,state['jitter'])
        assert self.jitter  == state['jitter'], err_m
    def loads(self,state):
        self.alpha = state['weights']
        self.trainer.loads(state['trainer'])
        self.jitter  = state['jitter']

class TrainerCholesky(TrainerBase):
    def __init__(self,memory_efficient):
        self.memory_eff = memory_efficient
    def fit(self,kernel,y,jitter):
        """ len(y) == len(reg)"""
        reg = jitter
        diag = kernel.diagonal().copy() 
        if self.memory_eff:
            np.fill_diagonal(kernel,np.add(diag, reg,out=diag))
            kernel, lower = cho_factor(kernel, lower=False, overwrite_a=True, check_finite=False)
            L = kernel
        else:
            L, lower = cho_factor(kernel + np.diag(reg), lower=False, overwrite_a=False,check_finite=False)
        alpha = cho_solve((L, lower), y ,overwrite_b=False).reshape((1,-1))
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
    
    def __init__(self,jitter,cv):
        self.jitter = jitter
        self.cv = cv
    
    def fit(self,kernel,y):
        '''Train the krr model with trainKernel and trainLabel.'''
        Q = kernel + np.diag(self.jitter*np.ones(kernel.shape[0]))
        Q_inv = np.linalg.inv(Q)
        alpha = np.dot(Q_inv,y)
        Cii = []
        beta = np.zeros(alpha.shape)
        self.y_pred = np.zeros(y.shape)
        self.error = np.zeros(y.shape)
        for _,test in self.cv.split(kernel):
            Cii = Q_inv[np.ix_(test,test)]
            beta = np.linalg.solve(Cii,alpha[test])
            self.y_pred[test] = y[test] - beta
            self.error[test] = beta
        
    def predict(self,kernel=None):
        '''kernel.shape is expected as (nPred,nTrain)'''
        #kernel = self.kernel(X,self.X_train)
        return self.y_pred

    def get_params(self,deep=True):
        return dict(sigma=self.jitter ,cv=self.cv)
        
    def set_params(self,params,deep=True):
        self.jitter  = params['jitter']
        self.cv = params['cv']
        self.y_pred = None

    def pack(self):
        state = dict(y_pred=self.y_pred,cv=self.cv.pack(),
                     jitter=self.jitter )
        return state
    def unpack(self,state):
        self.y_pred = state['y_pred']
        self.cv.unpack(state['cv'])
        err_m = 'jitter are not consistent {} != {}'.format(self.jitter ,state['jitter'])
        assert self.jitter  == state['jitter'], err_m
    def loads(self,state):
        self.y_pred = state['y_pred']
        self.cv.loads(state['cv'])
        self.jitter  = state['jitter']