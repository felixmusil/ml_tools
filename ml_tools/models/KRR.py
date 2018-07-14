import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular,cho_factor
from ..base import TrainerBase,RegressorBase


class KRR(RegressorBase):
    _pairwise = True
    
    def __init__(self,sigma,trainer):
        # Weights of the krr model
        self.alpha = None
        self.sigma = sigma
        self.trainer = trainer
    
    def fit(self,kernel,y):
        '''Train the krr model with trainKernel and trainLabel.'''
        #self.X_train = X
        #kernel = self.kernel(X)
        diag = kernel.diagonal().copy()
        reg = np.ones(diag.shape)*np.divide(np.multiply(self.sigma ** 2, np.mean(diag)), np.var(y))
        self.regularization = reg
        self.alpha = self.trainer.fit(kernel,y,regularization=reg)
        
    def predict(self,kernel):
        '''kernel.shape is expected as (nPred,nTrain)'''
        #kernel = self.kernel(X,self.X_train)
        return np.dot(kernel,self.alpha.flatten()).reshape((-1))
    def get_params(self,deep=True):
        return dict(sigma=self.sigma,trainer=self.trainer)
        
    def set_params(self,params,deep=True):
        self.sigma = params['sigma']
        self.trainer = params['trainer']
        self.alpha = None

    def pack(self):
        state = dict(weights=self.alpha,trainer=self.trainer.pack(),
                     regularization=self.regularization,sigma=self.sigma)
        return state
    def unpack(self,state):
        self.alpha = state['weights']
        self.trainer.unpack(state['trainer'])
        self.regularization = state['regularization']
        err_m = 'sigma are not consistent {} != {}'.format(self.sigma,state['sigma'])
        assert self.sigma == state['sigma'], err_m
    def loads(self,state):
        self.alpha = state['weights']
        self.trainer.loads(state['trainer'])
        self.regularization = state['regularization']
        self.sigma = state['sigma']

class TrainerCholesky(TrainerBase):
    def __init__(self,memory_efficient):
        self.memory_eff = memory_efficient
    def fit(self,kernel,y,regularization):
        """ len(y) == len(reg)"""
        reg = regularization
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