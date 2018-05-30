import numpy as np
from scipy.linalg import cho_factor,cho_solve
from ..utils import make_new_dir
import os 

def dummy(a):
    return a

def unpack_power_kernel_params(kernel_params):
    zeta = kernel_params[0]
    return zeta

def get_power_kernel(X1,X2=None,kernel_params=None):
    zeta = unpack_power_kernel_params(kernel_params)
    if X2 is None:
        X2 = X1
    kernel = np.empty((X1.shape[0],X2.shape[0]))
    np.dot(X1,X2.T,out=kernel)
    np.power(kernel,zeta,out=kernel)
    return kernel

class KRR(object):
    def __init__(self,model_params=None,kernel_params=None,save_intermediates=False,
                      func=None,invfunc=None,memory_eff=False,dirname=None):
        if dirname is None:
            self.model_params = model_params
            self.kernel_params = kernel_params
            if func is None or invfunc is None:
                self.func = dummy
                self.invfunc = dummy
            else:
                self.func = func
                self.invfunc = invfunc
            # Weights of the krr model
            self.alpha = None
            self.memory_eff = memory_eff
        else:
            self.load(dirname)

    def fit(self,feature_matrix,labels,kernel_func=None,kernel=None):
        '''Train the krr model with feature matrix and trainLabel.'''
        if kernel_func is None:
            kernel_func = get_power_kernel
        
        if kernel is None:
            self.memory_eff = True
            kernel = kernel_func(feature_matrix,**self.kernel_params)
        # learn a function of the label
        trainLabel = self.func(labels)

        diag = kernel.diagonal().copy()
        self.lower = False
        reg = np.multiply(np.multiply(self.sigma ** 2, np.mean(diag)), np.var(trainLabel))
        # Effective regularization
        self.reg = reg
        if self.memory_eff:
            # kernel is modified here
            np.fill_diagonal(np.power(kernel, self.zeta, out=kernel),
                                np.add(np.power(diag,self.zeta,out=diag), reg,out=diag))

            kernel, lower = cho_factor(kernel, lower=self.lower, overwrite_a=True, check_finite=False)
            L = kernel
        else:
            # kernel is not modified here
            reg = np.diag(reg)
            L, lower = cho_factor(np.power(kernel, self.zeta) + reg, lower=self.lower, overwrite_a=False,check_finite=False)

        # set the weights of the krr model
        self.alpha = cho_solve((L, lower), trainLabel,overwrite_b=False).reshape((1,-1))

    def predict(self,kernel):
        '''kernel.shape is expected as (nTrain,nPred)'''
        if self.memory_eff:
            # kernel is modified in place here
            return self.invfunc(np.dot(self.alpha, np.power(kernel,self.zeta,out=kernel) ) ).reshape((-1))
        else:
            # kernel is not modified here
            return self.invfunc(np.dot(self.alpha, np.power(kernel,self.zeta) ) ).reshape((-1))
      
    def save(self,fn):
        dirname = os.path.dirname(fn)
        new_dirname = make_new_dir(dirname)
        np.save(new_dirname+'/params.npy',dict(func=self.func,invfunc=self.invfunc,self.model_params,
                                             memory_eff=self.memory_eff,kernel_params=self.kernel_params))
        np.save(new_dirname+'/weights.npy',self.alpha)
        
        
    def load(self,dirname):
        params = np.load(dirname+'params.npy').item()
        self.func,self.invfunc,self.memory_eff,self.kernel_params,self.model_params = \
            params['func'],params['invfunc'],params['memory_eff'],params['kernel_params'],params['model_params']
        self.zeta = self.kernel_params[0]
        self.sigma = self.model_params[0]
        self.alpha = np.load(dirname+'weights.npy')