# -*- coding: utf-8 -*-
from ..base import np,sp
# import numpy as np
# import scipy as sp

from ..base import BaseEstimator,TransformerMixin
from scipy.sparse import lil_matrix,csr_matrix,issparse
from ..utils import tqdm_cs,s2hms
from time import time
from ..math_utils import symmetrize,get_unlin_soap
import gc


class CompressorCovarianceUmat(BaseEstimator,TransformerMixin):
    def __init__(self,soap_params=None,compression_type='species',compression_idx=5,scaling_weights=None,stride_size=None,
                    symmetric=False,to_reshape=True,normalize=True,dtype='float64',optimize_feature=False):

        self.compression_type = compression_type
        self.soap_params = soap_params
        self.symmetric = symmetric
        self.to_reshape = to_reshape
        self.normalize = normalize
        self.dtype = dtype
        self.stride_size = stride_size

        if 'angular+' in self.compression_type or 'species+' in self.compression_type:
            self.is_relative_scaling = True
        else:
            self.is_relative_scaling = False

        self.optimize_feature = optimize_feature
        if optimize_feature is False:
            self.scaling_weights = None
            self.feature_scaling = False
            self.full_opt = None
        elif optimize_feature == 'full':
            self.feature_scaling = True
            self.full_opt = True
        elif optimize_feature == 'diag':
            self.feature_scaling = True
            self.full_opt = False

        self.set_scaling_weights(scaling_weights)

        # number of elements to keep
        self.compression_idx = compression_idx

    def get_params(self,full_state=False,deep=True):
        params = dict(compression_idx=self.compression_idx,compression_type=self.compression_type,
                      soap_params=self.soap_params,scaling_weights=self.scaling_weights,
                      optimize_feature=self.optimize_feature,
                      dtype=self.dtype,stride_size=self.stride_size,
                      symmetric=self.symmetric,to_reshape=self.to_reshape,
                      normalize=self.normalize)

        if full_state is True:
            params['scale_relative_features_str'] = self.scale_relative_features_str
            params['feature_scaling'] = self.feature_scaling
            params['relative_scaling_weights'] = self.relative_scaling_weights
            params['scale_features_diag_str'] = self.scale_features_diag_str
            params['full_opt'] = self.full_opt
            params['scale_features_full_str'] = self.scale_features_full_str
            params['projection_str'] = self.projection_str
        return params

    def set_params(self,params,full_state=False):
        self.compression_idx = params['compression_idx']
        self.compression_type = params['compression_type']
        self.soap_params = params['soap_params']
        self.scaling_weights = params['scaling_weights']
        self.dtype = params['dtype']
        self.symmetric = params['symmetric']
        self.to_reshape = params['to_reshape']
        self.normalize = params['normalize']
        self.stride_size = params['stride_size']
        self.optimize_feature = params['optimize_feature']

        if full_state is True:
            self.scale_relative_features_str = str(params['scale_relative_features_str'])
            self.feature_scaling = params['feature_scaling']
            self.scale_features_full_str = str(params['scale_features_full_str'])
            self.relative_scaling_weights = params['relative_scaling_weights']
            self.scale_features_diag_str = str(params['scale_features_diag_str'])
            self.full_opt = params['full_opt']
            self.projection_str = str(params['projection_str'])

        nspecies = len(self.soap_params['global_species'])
        lmax1 = self.soap_params['lmax'] + 1
        nmax = self.soap_params['nmax']
        identity = lambda x: x
        reshape = lambda x: x.transpose(0,1,3,2,4,5).reshape((-1,nspecies*nmax, nspecies*nmax,lmax1))
        reshape_1 = lambda x: x.reshape((-1,nspecies*nspecies, nmax,nmax,lmax1))

        if 'angular+' in self.compression_type or 'species+' in self.compression_type:
            self.is_relative_scaling = True
        else:
            self.is_relative_scaling = False

        if 'species*radial' in self.compression_type:
            self.modify = reshape
        elif 'species+radial' in self.compression_type:
            self.modify = reshape_1
        else:
            self.modify = identity

    def pack(self):
        params = self.get_params(full_state=True)
        data = dict(u_mat_full=self.u_mat_full.tolist(),
                    eig=self.eig.tolist())
        state = dict(data=data,
                     params=params)
        return state

    def unpack(self,state):
        self.set_params(state['params'],full_state=True)
        self.u_mat_full = np.array(state['data']['u_mat_full'])
        self.eig = np.array(state['data']['eig'])

    def set_scaling_weights(self,factors):
        if factors is None:
            self.relative_scaling_weights = None
            self.scaling_weights = None
            factors__ = None
        else:

            factors_ = np.array(factors, self.dtype)

            if 'angular+' in self.compression_type:
                self.relative_scaling_weights = factors_[:int(self.soap_params['lmax']+1)]
                factors__ = factors_[int(self.soap_params['lmax']+1):]
            elif 'species+' in self.compression_type:
                Nsp = len(self.soap_params['global_species'])
                self.relative_scaling_weights = factors_[:Nsp]
                factors__ = factors_[Nsp:]
            else:
                self.relative_scaling_weights = None
                factors__ = factors_

        if self.full_opt is False:
            self._set_diagonal_scaling_weights(factors__)
        elif self.full_opt is True:
            self._set_full_scaling_weights(factors__)

    def _set_diagonal_scaling_weights(self,x):
        if x is None:
            self.scaling_weights = None
            self.compression_idx = None
        else:
            self.scaling_weights = x.flatten()
            self.compression_idx = len(self.scaling_weights)

    def _set_full_scaling_weights(self,u_mat):
        if u_mat is None:
            self.scaling_weights = None
            self.compression_idx = None
        else:
            self.compression_idx = int(np.sqrt(np.array(u_mat).size))
            self.scaling_weights = np.array(u_mat).reshape((self.compression_idx,self.compression_idx))

    def reshape_(self,X):
        unwrapped_X = get_unlin_soap(X,self.soap_params,
                        self.soap_params['global_species'],dtype=self.dtype)
        return unwrapped_X

    def get_covariance_umat_full(self,X):
        u_mat_full, eig = self._get_covariance_umat_full(X)
        return u_mat_full, eig

    def project(self,X):
        stride_size = self.stride_size
        N = X.shape[0]

        bounds = self._get_bounds(N, stride_size)

        test = self._project(X[0:1])
        M = test.shape[1]

        X_compressed = np.ones((N,M),self.dtype)
        for st,nd in tqdm_cs(bounds,desc='Umat Proj',leave=False):
            X_c = X[st:nd]
            X_c = self._project(X_c)
            X_compressed[st:nd] = X_c

        return X_compressed

    def scale_features(self,unlin_soap,stride_size=None):
        if stride_size is None:
            stride_size = self.stride_size
        N = unlin_soap.shape[0]

        bounds = self._get_bounds(N, stride_size)

        X_compressed = []
        for st,nd in tqdm_cs(bounds,desc='Scale Feat',leave=False):
            X_c = unlin_soap[st:nd]
            X_c = self._scale_features(X_c)
            X_c = self._prepare_output(X_c,lin_out=True)
            Nsoap = X_c.shape[0]
            X_compressed.append(X_c.reshape((Nsoap,-1)))

        if stride_size is None:
            return X_compressed[0]
        elif stride_size is not None:
            aa = np.concatenate(X_compressed,axis=0)
        return aa

    def project_on_u_mat(self,X):
        N = X.shape[0]
        bounds = self._get_bounds(N, self.stride_size)

        test = self._prepare_input(X[0:1])
        test = self._get_compressed_soap(test)

        shape = [N] + list(test.shape[1:])
        X_compressed = np.ones(shape,self.dtype)

        for st,nd in tqdm_cs(bounds,desc='Umat Proj',leave=False):
            X_c = X[st:nd]
            X_c = self._prepare_input(X_c)
            X_c = self._get_compressed_soap(X_c)
            X_compressed[st:nd] = X_c

        return X_compressed

    def fit(self,X):
        self.u_mat_full, self.eig = self.get_covariance_umat_full(X)
        return self

    def transform(self,X):
        X_proj = self.project(X)
        return X_proj

    def fit_transform(self,X):
        return self.fit(X).transform(X)

    def _get_covariance_umat_full(self,rawsoaps):
        '''
        Compute the covariance of the given unlinsoap and decomposes it
        unlinsoap.shape = (Nsample,nspecies, nspecies, nmax, nmax, lmax+1)
        '''

        X_c = rawsoaps.mean(axis=0).reshape((1,-1))

        X_c = self._prepare_input(X_c,to_modify=False)

        X = np.squeeze(X_c)

        nspecies, nspecies, nmax, nmax, lmax1 =  X.shape

        l_factor = np.sqrt(2*np.arange(lmax1)+1)
        X *= l_factor.reshape((1,1,1,1,-1))

        identity = lambda x: x
        reshape = lambda x: x.transpose(0,1,3,2,4,5).reshape((-1,nspecies*nmax, nspecies*nmax,lmax1))
        reshape_1 = lambda x: x.reshape((-1,nspecies*nspecies, nmax,nmax,lmax1))
        if self.compression_type in ['species']:
            cov = X.mean(axis=(4)).trace(axis1=2, axis2=3) / nmax
            self.projection_str = 'ij,nm,ajmopl->ainopl'
            self.scale_features_diag_str = 'j,m,ajmopl->ajmopl'
            self.scale_features_full_str = self.projection_str
            self.scale_relative_features_str = None
            self.modify = identity
        elif self.compression_type in ['angular+species']:
            X_c = X.transpose(4,0,1,2,3)
            cov = X_c.trace(axis1=3, axis2=4) / nmax
            self.projection_str = 'lij,lnm,ajmopl->ainopl'
            self.scale_features_diag_str = 'j,m,ajmopl->ajmopl'
            self.scale_features_full_str = 'ij,nm,ajmopl->ainopl'
            self.scale_relative_features_str = 'l,ajmopl->ajmopl'
            self.modify = identity
        elif self.compression_type in ['species+radial']:
            X_c = X.mean(axis=(4)).reshape((-1, nmax, nmax))
            cov = X_c
            self.projection_str = 'oij,onm,aojml->aoinl'
            self.scale_features_diag_str = 'j,m,paojml->paojml'
            self.scale_features_full_str = 'ij,nm,paojml->paoinl'
            self.scale_relative_features_str = 'a,o,paojml->paojml'
            self.modify = reshape_1
        elif self.compression_type in ['radial']:
            cov = X.mean(axis=(4)).trace(axis1=0, axis2=1) / nspecies
            self.projection_str = 'ij,nm,aopjml->aopinl'
            self.scale_features_diag_str = 'j,m,aopjml->aopjml'
            self.scale_features_full_str = 'ij,nm,aopjml->aopinl'
            self.scale_relative_features_str = None
            self.modify = identity
        elif self.compression_type in ['angular+radial']:
            X_c = X.transpose(4,0,1,2,3)
            cov = X_c.trace(axis1=1, axis2=2) / nspecies
            self.projection_str = 'lij,lnm,aopjml->aopinl'
            self.scale_features_diag_str = 'j,m,aopjml->aopjml'
            self.scale_features_full_str = 'ij,nm,aopjml->aopinl'
            self.scale_relative_features_str = 'l,aopjml->aopjml'
            self.modify = identity
        elif self.compression_type in ['species*radial']:
            X_c = X.transpose(0,2,1,3,4).reshape((nspecies*nmax, nspecies*nmax,lmax1))
            cov = X_c.mean(axis=(2))
            self.projection_str = 'ij,nm,ajml->ainl'
            self.scale_features_diag_str = 'j,m,ajml->ajml'
            self.scale_features_full_str = 'ij,nm,ajml->ainl'
            self.scale_relative_features_str = None
            # there is a contraction here so we need to reshape the input before transforming it
            self.modify = reshape
        elif self.compression_type in ['angular+species*radial']:
            X_c = X.transpose(4,0,2,1,3).reshape((lmax1,nspecies*nmax, nspecies*nmax))
            cov = X_c
            self.projection_str = 'lij,lnm,ajml->ainl'
            self.scale_features_diag_str = 'j,m,ajml->ajml'
            self.scale_features_full_str = 'ij,nm,ajml->ainl'
            self.scale_relative_features_str = 'l,ajml->ajml'
            self.modify = reshape

        # get sorted eigen values and vectors
        eva,eve = np.linalg.eigh(cov)
        # first dim is angular or species**2 so the flip has to be shifted
        if self.is_relative_scaling is True:
            eva = np.flip(eva,axis=1)
            eve = np.flip(eve,axis=2)
            return eve.transpose((0,2,1)),eva
        else:
            eva = np.flip(eva,axis=0)
            eve = np.flip(eve,axis=1)
            return eve.T,eva.flatten()

    def _scale_features(self,X):
        X_c = self._relative_scaling(X)
        X_c = self._umat_scaling(X_c)
        return X_c

    def _project(self,X):
        X_c = self._prepare_input(X)
        X_c = self._get_compressed_soap(X_c)

        if self.feature_scaling is True:
            X_c = self._scale_features(X_c)

        X_c = self._prepare_output(X_c,lin_out=True)

        return X_c

    def _get_compressed_soap(self,unlinsoap):
        kwargs = dict(optimize='optimal')
        # kwargs = dict(optimize=False)

        if self.is_relative_scaling is True:
            u_mat = np.array(self.u_mat_full[:,:self.compression_idx,:],dtype=self.dtype)
        else:
            u_mat = np.array(self.u_mat_full[:self.compression_idx,:],dtype=self.dtype)

        X_c = np.einsum(self.projection_str,u_mat,u_mat,unlinsoap,**kwargs)
        # X_c = get_compressed_soap(X,u_mat,self.projection_str)

        if self.compression_type in ['species+radial']:
            Nsoap, Nsp2, nmax, nmax, lmax1 = X_c.shape
            nspecies = int(np.sqrt(Nsp2))
            X_c = X_c.reshape((Nsoap, nspecies, nspecies, nmax, nmax, lmax1))

        return X_c

    def _prepare_input(self,X,to_modify=True):

        if issparse(X) is True:
            X_c = np.asarray(X.toarray(),dtype=self.dtype)
        else:
            X_c = np.asarray(X,dtype=self.dtype)

        if self.to_reshape is True:
            X_c = self.reshape_(X_c)

        if to_modify is True:
            X_c = self.modify(X_c)

        return X_c

    def _prepare_output(self, X, symmetric=None, normalize=None, lin_out=None):
        if symmetric is None:
            symmetric = self.symmetric
        if normalize is None:
            normalize = self.normalize
        if lin_out is None:
            lin_out = self.lin_out

        p = transform_feature_matrix(X, symmetric=symmetric, normalize=normalize, lin_out=lin_out)

        return p

    def _relative_scaling(self,rawsoap_unlin):
        kwargs = dict(optimize='optimal')
        # kwargs = dict(optimize=False)

        if self.is_relative_scaling is True:
            args = [self.scale_relative_features_str]

            if self.compression_type in ['species+radial']:
                args += [self.relative_scaling_weights, self.relative_scaling_weights, rawsoap_unlin]
            else:
                args += [self.relative_scaling_weights, rawsoap_unlin]

            X_c = np.einsum(*args,**kwargs)
            return X_c
        else:
            return rawsoap_unlin

    def _umat_scaling(self,rawsoap_unlin):
        if self.full_opt is False:
            scale_features_str = self.scale_features_diag_str
        elif self.full_opt is True:
            scale_features_str = self.scale_features_full_str

        kwargs = dict(optimize='optimal')
        # kwargs = dict(optimize=False)
        args = [scale_features_str]
        args += [self.scaling_weights,self.scaling_weights,rawsoap_unlin]

        p = np.einsum(*args,**kwargs)
        return p

    def _get_bounds(self,N,stride_size=None):
        if stride_size is None:
            bounds = [(0,N)]
        elif stride_size is not None:
            Nstride = N // stride_size
            bounds = [[ii*stride_size,(ii+1)*stride_size] for ii in range(Nstride)]
            bounds[-1][-1] += N % stride_size
        return bounds


class AngularScaler(BaseEstimator,TransformerMixin):
    def __init__(self,soap_params=None,fj=None,to_reshape=True):
        self.soap_params = soap_params
        self.fj = fj
        self.to_reshape = to_reshape

    def get_params(self,deep=True):
        params = dict(soap_params=self.soap_params,fj=self.fj,
                      to_reshape=self.to_reshape)
        return params

    def set_params(self,params):
        self.soap_params = params['soap_params']
        self.fj = params['fj']
        self.to_reshape = params['to_reshape']

    def set_fj(self,fj):
        #self.u_mat = None
        self.fj = fj

    def reshape_(self,X):
        unwrapped_X = get_unlin_soap(X,self.soap_params,self.soap_params['global_species'])

        Nsample,nspecies, nspecies, nmax, nmax, lmax1 =  unwrapped_X.shape

        X_c = unwrapped_X.transpose(0,1,3,2,4,5).reshape((Nsample,nspecies*nmax, nspecies*nmax,lmax1))
        unwrapped_X = None
        return symmetrize(X_c)

    def fit(self,X=None):
        return self

    def transform(self,X):
        if self.to_reshape:
            X_c = self.reshape_(X)
        else:
            X_c = X

        Nsoap,_,_ = X_c.shape
        aa = np.einsum("ijk,k->ijk",X_c,self.fj,optimize='optimal').reshape((Nsoap,-1))
        #aa = np.tensordot(X_c,np.diag(self.fj),axes=(2,0))
        aan = np.linalg.norm(aa,axis=1).reshape((Nsoap,1))
        aa /= aan

        return aa

    def fit_transform(self,X):
        return self.transform(X)

    def pack(self):
        params = self.get_params()
        state = dict(params=params)
        return state

    def unpack(self,state):
        self.set_params(state['params'])


def transform_feature_matrix(p, symmetric=False, normalize=True, lin_out=True):
    dtype = p.dtype
    Ndim = len(p.shape)

    if Ndim == 6:
        Nsoap,nspecies, nspecies, nmax, nmax, lmax1 = p.shape
        shape1 = (Nsoap,1,1,1,1,1)
        trans = True
    elif Ndim == 4:
        Nsoap, Ncomp , Ncomp , lmax1 = p.shape
        shape1 = (Nsoap,1,1,1)
        trans = False

    if normalize is True:
        pn = np.linalg.norm(p.reshape((Nsoap,-1)),axis=1).reshape(shape1)
        p /=  pn

    if symmetric is True:
        p = p.transpose(0,1,3,2,4,5).reshape(Nsoap,nspecies*nmax, nspecies*nmax, lmax1) if trans is True else p
        p = symmetrize(p,dtype)

    if lin_out:
        p = p.reshape((Nsoap,-1))

    return p
