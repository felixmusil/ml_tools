# -*- coding: utf-8 -*-

from ..base import np,sp
from ..base import BaseEstimator,TransformerMixin
from ..utils import tqdm_cs
import matplotlib.pyplot as plt

class FPSFilter(BaseEstimator,TransformerMixin):
    def __init__(self,Nselect,kernel,act_on='sample',precompute_kernel=True,disable_pbar=True):
        self.Nselect = Nselect
        if isinstance(kernel, np.ndarray) is True or isinstance(kernel,np.memmap) is True:
            # give a matrix kernel
            self.precompute_kernel = None
        else:
            # give a function or class kernel
            self.precompute_kernel = precompute_kernel
        self.kernel = kernel
        self.disable_pbar = disable_pbar
        self.transformation_mat = None
        if act_on in ['sample','feature','feature A transform']:
            self.act_on = act_on  
        else:
            raise 'Wrong input: {}'.format(act_on)
        
    
    def get_params(self,deep=True):
        params = dict(Nselect=self.Nselect,kernel=self.kernel,act_on=self.act_on)
        return params
    
    def fit(self,X,dry_run=False):
        if isinstance(X,dict):
            x = X['feature matrix']
        else:
            x = X
        
        if dry_run:
            Nselect = x.shape[0]
        else:
            Nselect = self.Nselect
        
        if self.act_on in ['sample']:
            pass
        elif self.act_on in ['feature','feature A transform']:
            x = x.T
        
        
        if self.precompute_kernel is True:
            kernel = self.kernel(x)
            ifps, dfps = do_fps_kernel(kernel,d=Nselect,disable_pbar=self.disable_pbar)
        elif self.precompute_kernel is False:
            ifps, dfps = do_fps_feature(x,d=Nselect,kernel=self.kernel,disable_pbar=self.disable_pbar)
        elif self.precompute_kernel is None:
            ifps, dfps = do_fps_kernel(self.kernel,d=Nselect,disable_pbar=self.disable_pbar)
        
        if self.act_on == 'feature A transform':
            self.transformation_mat = get_A_matrix(x.T[:,ifps],x.T)
        
        self.selected_ids = ifps
        self.min_distance2 = dfps
        self.trained = True
        return self
    
    def transform(self,X):
        if self.act_on == 'sample' and self.trained is True:
            # Only the training set needs to be sparsified 
            # at prediction time it should do nothing to the 
            # new samples
            self.trained = False
            return X[self.selected_ids,:]
        elif self.act_on == 'feature':
            return X[:,self.selected_ids]
        elif self.act_on == 'feature A transform':
            return np.dot(X[:,self.selected_ids],self.transformation_mat)
        else:
            return X
        
    def plot(self):
        plt.semilogy(self.min_distance2,label='FPSFilter '+self.act_on)



def do_fps_feature(x=None, d=0,init=0,disable_pbar=True,kernel=None):
    if d == 0 : d = x.shape[0]
    n = x.shape[0]
    iy = np.zeros(d, int)
    # faster evaluation of Euclidean distance
    n2 = np.zeros((n,))
    for ii in range(n):
        n2[ii] = kernel(x[ii],x[ii])
    
    iy[0] = init
    dl = n2 + n2[iy[0]] - 2* kernel(x, x[iy[0]])
    lmin = np.zeros(d)
    for i in tqdm_cs(range(1,d),leave=False,desc='fps',disable=disable_pbar):
        iy[i] = np.argmax(dl)
        lmin[i-1] = dl[iy[i]]    
        nd = n2 + n2[iy[i]] - 2*kernel(x,x[iy[i]])
        dl = np.minimum(dl, nd)
    return iy, lmin

def do_fps_kernel(kernel, d=0,init=0,disable_pbar=True):
    if d == 0 : d = len(kernel)
    n = len(kernel)
    iy = np.zeros(d, int)
    # faster evaluation of Euclidean distance
    n2 = kernel.diagonal().copy()
    iy[0] = init
    dl = n2 + n2[iy[0]] - 2* kernel[:,iy[0]]
    lmin = np.zeros(d)
    for i in tqdm_cs(range(1,d),leave=False,desc='fps',disable=disable_pbar):
        iy[i] = np.argmax(dl)
        lmin[i-1] = dl[iy[i]]    
        nd = n2 + n2[iy[i]] - 2*kernel[:,iy[i]]
        dl = np.minimum(dl, nd)
    return iy, lmin


def get_A_matrix(Xfs,Xff):
    CpX=np.dot(np.linalg.pinv(Xfs),Xff)
    W = np.dot(CpX,CpX.T)
    eva,eve = np.linalg.eigh(W);
    A = np.dot(np.dot(eve,np.diag(np.sqrt(eva))),eve.T)
    return A
