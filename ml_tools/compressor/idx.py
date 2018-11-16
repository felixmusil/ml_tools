# -*- coding: utf-8 -*-
# Filter class that sparsifies features/a kernel based on
# a list of indices
# AAEAE 
from ..base import np,sp
from ..base import BaseEstimator,TransformerMixin
from ..utils import tqdm_cs
import matplotlib.pyplot as plt

class IDXFilter(BaseEstimator,TransformerMixin):
    def __init__(self,Iselect,kernel,act_on='sample',disable_pbar=True):
	self.Iselect = Iselect
        self.kernel = kernel
        self.disable_pbar = disable_pbar
        self.transformation_mat = None
        if act_on in ['sample','feature','feature A transform']:
            self.act_on = act_on  
        else:
            raise 'Wrong input: {}'.format(act_on)
        
    
    def get_params(self,deep=True):
        params = dict(kernel=self.kernel,act_on=self.act_on)
        return params
    
    def fit(self,X,dry_run=False):
        if isinstance(X,dict):
            x = X['feature matrix']
        else:
            x = X
        
        if self.act_on in ['sample']:
            pass
        elif self.act_on in ['feature','feature A transform']:
            x = x.T
        
        if self.act_on == 'feature A transform':
            self.transformation_mat = get_A_matrix(x.T[:,ifps],x.T)
        
        self.trained = True
        return self
    
    def transform(self,X,y):
        if self.act_on == 'sample' and self.trained is True:
            # Only the training set needs to be sparsified 
            # at prediction time it should do nothing to the 
            # new samples
            self.trained = False
            return X[self.Iselect,:],y[self.Iselect]
        elif self.act_on == 'feature':
            return X[:,self.Iselect],y
        elif self.act_on == 'feature A transform':
            return np.dot(X[:,self.Iselect],self.transformation_mat),y
        else:
            return X,y
        
def get_A_matrix(Xfs,Xff):
    CpX=np.dot(np.linalg.pinv(Xfs),Xff)
    W = np.dot(CpX,CpX.T)
    eva,eve = np.linalg.eigh(W);
    A = np.dot(np.dot(eve,np.diag(np.sqrt(eva))),eve.T)
    return A
