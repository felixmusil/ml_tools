from sklearn.base import BaseEstimator, RegressorMixin,TransformerMixin

try:
    import autograd.numpy as np
    import autograd.scipy as sp
except:
    import numpy as np
    import scipy as sp

class RegressorBase(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        pass
    def get_params(self,deep=True):
        pass
    def get_name(self):
        return type(self).__name__


class AtomicDescriptorBase(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        pass
    def get_params(self,deep=True):
        pass
    def get_name(self):
        return type(self).__name__

class TrainerBase(object):
    def __init__(self):
        pass
    def fit(self):
        pass


class KernelBase(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def get_params(self):
        pass
    def set_params(self,**params):
        pass
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the kernel."""
    def get_name(self):
        return type(self).__name__