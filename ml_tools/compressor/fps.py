# -*- coding: utf-8 -*-

from ..base import np,sp
from ..base import CompressorBase,FeatureBase
from ..utils import tqdm_cs
import matplotlib.pyplot as plt

class FPSFilter(CompressorBase):
    def __init__(self,Nselect,kernel,act_on='sample',precompute_kernel=True,disable_pbar=True):
        super(FPSFilter,self).__init__()

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

        self.transformation_mat = None
        self.selected_ids = None
        self.min_distance2 = None
        self.trained = False


    def get_params(self,deep=True):
        params = dict(Nselect=self.Nselect,kernel=self.kernel,act_on=self.act_on,precompute_kernel=self.precompute_kernel,
        disable_pbar=self.disable_pbar)
        return params

    def fit(self,X,dry_run=False):
        is_feature_selection = self.act_on in ['feature','feature A transform']
        if isinstance(X,dict) and is_feature_selection:
            x = X['feature_matrix'].T
        elif isinstance(X,FeatureBase):
            gradients = False
            if gradients is False:
                x = X

            if is_feature_selection is True:
                x = x.T
        elif is_feature_selection:
            x = X.T
        else:
            x = X



        if dry_run and isinstance(X,dict):
            Nselect = x['feature_matrix'].shape[0]
        elif dry_run:
            if isinstance(X,FeatureBase):
                Nselect = x.get_nb_sample(gradients)
            else:
                Nselect = x.shape[0]
        else:
            Nselect = self.Nselect

        if self.precompute_kernel is True:
            kernel = self.kernel.transform(x)
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
            if isinstance(X,FeatureBase):
                return X.extract_pseudo_input(self.selected_ids[:self.Nselect])
            else:
                return X[self.selected_ids[:self.Nselect],:]
        elif self.act_on == 'feature':
            if isinstance(X,FeatureBase):
                return X.extract_feature_selection(self.selected_ids[:self.Nselect])
            else:
                return X[:,self.selected_ids[:self.Nselect]]
        elif self.act_on == 'feature A transform':
            if isinstance(X,FeatureBase):
                X_selected = X.extract_feature_selection(self.selected_ids[:self.Nselect])
                X_selected.representations = np.dot(X_selected.representations,self.transformation_mat)
                return X_selected
            else:
                return np.dot(X[:,self.selected_ids[:self.Nselect]],self.transformation_mat)
        else:
            return X

    def plot(self):
        plt.semilogy(self.min_distance2,label='FPSFilter '+self.act_on)

    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(transformation_mat=self.transformation_mat,selected_ids=self.selected_ids,min_distance2=self.min_distance2,trained=self.trained)
        return state

    def loads(self,data):
        self.transformation_mat = data['transformation_mat']
        self.selected_ids = data['selected_ids']
        self.min_distance2 = data['min_distance2']
        self.trained = data['trained']


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
    eva,eve = np.linalg.eigh(W)
    A = np.dot(np.dot(eve,np.diag(np.sqrt(eva))),eve.T)
    return A