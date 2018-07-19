
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