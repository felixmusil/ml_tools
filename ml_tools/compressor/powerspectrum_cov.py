from ..base import np,sp
from ..base import BaseEstimator,TransformerMixin
from scipy.sparse import lil_matrix,csr_matrix,issparse
from ..utils import tqdm_cs,s2hms
from time import time
from ..math_utils import symmetrize,get_unlin_soap

class CompressorCovarianceUmat(BaseEstimator,TransformerMixin):
    def __init__(self,soap_params=None,compression_type='species',compression_idx=5,scaling_weights=None,stride_size=None,
                    symmetric=False,to_reshape=True,normalize=True,dtype='float64',optimize_feature=False):
        # number of elements to keep
        self.compression_idx = compression_idx
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

    def get_params(self,deep=True):
        params = dict(compression_idx=self.compression_idx,compression_type=self.compression_type,
                      soap_params=self.soap_params,scaling_weights=self.scaling_weights,
                      feature_scaling=self.feature_scaling,
                      relative_scaling_weights=self.relative_scaling_weights,
                      dtype=self.dtype,stride_size=self.stride_size,
                      scale_features_diag_str=self.scale_features_diag_str,
                      full_opt=self.full_opt,
                      scale_features_full_str=self.scale_features_full_str,
                      symmetric=self.symmetric,to_reshape=self.to_reshape,
                      projection_str=self.projection_str,normalize=self.normalize)
        return params

    def set_params(self,params):
        self.compression_idx = params['compression_idx']
        self.compression_type = params['compression_type']
        self.soap_params = params['soap_params']
        self.scaling_weights = params['scaling_weights']
        self.dtype = params['dtype']
        self.symmetric = params['symmetric']
        self.to_reshape = params['to_reshape']
        self.projection_str = params['projection_str']
        self.normalize = params['normalize']
        self.scale_features_diag_str = params['scale_features_diag_str']
        self.stride_size = params['stride_size']
        self.full_opt = params['full_opt']
        self.scale_features_full_str = params['scale_features_full_str']
        self.relative_scaling_weights = params['relative_scaling_weights']

        nspecies = len(self.soap_params['global_species'])
        lmax1 = self.soap_params['lmax'] + 1
        nmax = self.soap_params['nmax']
        identity = lambda x: x
        reshape = lambda x: x.transpose(0,1,3,2,4,5).reshape((-1,nspecies*nmax, nspecies*nmax,lmax1))
        reshape_1 = lambda x: x.reshape((-1,nspecies*nspecies, nmax,nmax,lmax1))

        if 'species*radial' in self.compression_type:
            self.modify = reshape
        elif 'species+radial' in self.compression_type:
            self.modify = reshape_1
        else:
            self.modify = identity

    def set_scaling_weights(self,factors):
        factors = np.array(factors, self.dtype)

        if 'angular+' in self.compression_type:
            self.relative_scaling_weights = factors[:int(self.soap_params['lmax']+1)]
            factors_ = factors[int(self.soap_params['lmax']+1):]
        elif 'species+' in self.compression_type:
            self.relative_scaling_weights = factors[:int(self.soap_params['global_species']**2)]
            factors_ = factors[int(self.soap_params['global_species']**2):]
        else:
            factors_ = factors

        if self.full_opt is False:
            self.set_diagonal_scaling_weights(factors_)
        elif self.full_opt is True:
            self.set_full_scaling_weights(factors_)

    def set_diagonal_scaling_weights(self,x):
        self.scaling_weights = x.flatten()
        self.compression_idx = len(self.scaling_weights)

    def set_full_scaling_weights(self,u_mat):
        self.compression_idx = int(np.sqrt(np.array(u_mat).size))
        self.scaling_weights = np.array(u_mat).reshape((self.compression_idx,self.compression_idx))

    def reshape_(self,X):
        unwrapped_X = get_unlin_soap(X,self.soap_params,
                        self.soap_params['global_species'],dtype=self.dtype)
        return unwrapped_X

    def get_covariance_umat_full(self,unlinsoap):
        '''
        Compute the covariance of the given unlinsoap and decomposes it
        unlinsoap.shape = (Nsample,nspecies, nspecies, nmax, nmax, lmax+1)
        '''
        Nsample = unlinsoap.shape[0]

        if issparse(unlinsoap) is True:
            m_soap = np.array(unlinsoap.mean(axis=0)).reshape((1,-1))
            X = get_unlin_soap(m_soap,self.soap_params,
                            self.soap_params['global_species'],dtype=self.dtype)
        else:
            X = unlinsoap.mean(axis=0)

        X = np.squeeze(X)

        nspecies, nspecies, nmax, nmax, lmax1 =  X.shape

        l_factor = np.sqrt(2*np.arange(lmax1)+1)
        X *= l_factor.reshape((1,1,1,1,-1))

        identity = lambda x: x
        reshape = lambda x: x.transpose(0,1,3,2,4,5).reshape((-1,nspecies*nmax, nspecies*nmax,lmax1))
        reshape_1 = lambda x: x.reshape((-1,nspecies*nspecies, nmax,nmax,lmax1))
        if self.compression_type in ['species']:
            cov = X.mean(axis=(4)).trace(axis1=2, axis2=3)
            self.projection_str = 'ij,nm,ajmopl->ainopl'
            self.scale_features_diag_str = 'j,m,ajmopl->ajmopl'
            self.scale_features_full_str = self.projection_str
            self.scale_relative_features_str = None
            self.modify = identity
        elif self.compression_type in ['angular+species']:
            X_c = X.transpose(4,0,1,2,3)
            cov = X_c.trace(axis1=3, axis2=4)
            self.projection_str = 'lij,lnm,ajmopl->ainopl'
            self.scale_features_diag_str = 'j,m,ajmopl->ajmopl'
            self.scale_features_full_str = 'ij,nm,ajmopl->ainopl'
            self.scale_relative_features_str = 'l,ajmopl->ajmopl'
            self.modify = identity
        elif self.compression_type in ['species+radial']:
            X_c = X.mean(axis=(4)).reshape((-1, nmax, nmax))
            cov = X_c
            self.projection_str = 'oij,onm,aojml->aoinl'
            self.scale_features_diag_str = 'j,m,aojml->aojml'
            self.scale_features_full_str = 'ij,nm,aojml->aoinl'
            self.scale_relative_features_str = 'o,aojml->aojml'
            self.modify = reshape_1
        elif self.compression_type in ['radial']:
            cov = X.mean(axis=(4)).trace(axis1=0, axis2=1)
            self.projection_str = 'ij,nm,aopjml->aopinl'
            self.scale_features_diag_str = 'j,m,aopjml->aopjml'
            self.scale_features_full_str = 'ij,nm,aopjml->aopinl'
            self.scale_relative_features_str = None
            self.modify = identity
        elif self.compression_type in ['angular+radial']:
            X_c = X.transpose(4,0,1,2,3)
            cov = X_c.trace(axis1=1, axis2=2)
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

    def fit(self,X):
        if issparse(X) is True:
            X_c = X
        elif self.to_reshape is True:
            X_c = self.reshape_(np.asarray(X,dtype=self.dtype))
        else:
            X_c = np.asarray(X,dtype=self.dtype)

        self.u_mat_full, self.eig = self.get_covariance_umat_full(X_c)

        return self

    def project_on_u_mat(self,X,compression_only=False,stride_size=None):
        if stride_size is None:
            stride_size = self.stride_size
        N = X.shape[0]

        if issparse(X) is False and stride_size is None:
            bounds = [(0,N)]
        elif issparse(X) is True or stride_size is not None:
            Nstride = N // stride_size
            bounds = [[ii*stride_size,(ii+1)*stride_size] for ii in range(Nstride)]
            bounds[-1][-1] += N % stride_size

        X_compressed = []

        for st,nd in tqdm_cs(bounds,desc='Umat Proj',leave=False):

            if issparse(X) is True:
                X_ = np.asarray(X[st:nd].toarray(),dtype=self.dtype)
            else:
                X_ = np.asarray(X[st:nd],dtype=self.dtype)

            if self.to_reshape:
                X_c = self.modify(self.reshape_(X_))
            else:
                X_c = self.modify(X_)

            if self.is_relative_scaling is True:
                u_mat = np.array(self.u_mat_full[:,:self.compression_idx,:],dtype=self.dtype)
            else:
                u_mat = np.array(self.u_mat_full[:self.compression_idx,:],dtype=self.dtype)

            print(u_mat.shape,X_c.shape)
            if compression_only is False:
                X_compressed.append(get_compressed_soap(X_c,u_mat,self.projection_str,symmetric=False,
                                    lin_out=False,normalize=self.normalize))
            elif compression_only is True:
                X_compressed.append(get_compressed_soap(X_c,u_mat,self.projection_str,symmetric=True,
                                    lin_out=True,normalize=self.normalize))

        if issparse(X) is False and stride_size is None:
            return X_compressed[0]
        elif issparse(X) is True or stride_size is not None:
            aa = np.array(np.concatenate(X_compressed,axis=0),dtype=self.dtype)
            return aa

    def scale_relative_feature(self, projected_unlinsoap, stride_size=None):
        if stride_size is None:
            stride_size = self.stride_size

        N = projected_unlinsoap.shape[0]
        X_compressed = []
        bounds = self.get_bounds(N, stride_size)
        for st,nd in tqdm_cs(bounds,desc='Relative Feature Scaling',leave=False):
            args = [self.scale_relative_features_str]
            X_c = projected_unlinsoap[st:nd]
            args += [self.relative_scaling_weights, X_c]

            kwargs = dict(optimize='optimal')
            p = np.einsum(*args,**kwargs)

            Nsoap = p.shape[0]
            X_compressed.append(p.reshape((Nsoap,-1)))

        if stride_size is None:
            return X_compressed[0]
        elif stride_size is not None:
            aa = np.concatenate(X_compressed,axis=0)
            return aa

    def scale_features(self,projected_unlinsoap, stride_size=None):
        if stride_size is None:
            stride_size = self.stride_size

        if self.is_relative_scaling is True:
            projected_unlinsoap_ = self.scale_relative_feature(projected_unlinsoap, stride_size)
        else:
            projected_unlinsoap_ = projected_unlinsoap

        if self.full_opt is False:
            scale_features_str = self.scale_features_diag_str
        elif self.full_opt is True:
            scale_features_str = self.scale_features_full_str

        return self.scale_features_impl(projected_unlinsoap_,scale_features_str,stride_size)

    def get_bounds(self,N,stride_size=None):
        if stride_size is None:
            bounds = [(0,N)]
        elif stride_size is not None:
            Nstride = N // stride_size
            bounds = [[ii*stride_size,(ii+1)*stride_size] for ii in range(Nstride)]
            bounds[-1][-1] += N % stride_size
        return bounds

    def scale_features_impl(self,projected_unlinsoap,scale_features_str,stride_size=None):
        if stride_size is None:
            stride_size = self.stride_size

        N = projected_unlinsoap.shape[0]
        X_compressed = []
        bounds = self.get_bounds(N, stride_size)
        for st,nd in tqdm_cs(bounds,desc='Feature Scaling',leave=False):
            args = [scale_features_str]
            X_c = projected_unlinsoap[st:nd]

            args += [self.scaling_weights,self.scaling_weights,X_c]

            kwargs = dict(optimize='optimal')
            p = np.einsum(*args,**kwargs)

            p = transform_feature_matrix(p, symmetric=self.symmetric, normalize=self.normalize, lin_out=True)

            X_compressed.append(p)

        if stride_size is None:
            return X_compressed[0]
        elif stride_size is not None:
            aa = np.concatenate(X_compressed,axis=0)
            return aa

    def transform(self,X):
        if self.feature_scaling is True:
            X_proj = self.project_on_u_mat(X,compression_only=False,stride_size=self.stride_size)
            X_proj = self.scale_features(X_proj)
        else:
            X_proj = self.project_on_u_mat(X,compression_only=True,stride_size=self.stride_size)

        return X_proj

    def fit_transform(self,X):
        return self.fit(X).transform(X)

    def pack(self):
        params = self.get_params()
        data = dict(u_mat_full=self.u_mat_full.tolist(),
                    eig=self.eig.tolist())
        state = dict(data=data,
                     params=params)
        return state

    def unpack(self,state):
        self.set_params(state['params'])
        self.u_mat_full = np.array(state['data']['u_mat_full'])
        self.eig = np.array(state['data']['eig'])

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


def get_compressed_soap(unlinsoap,u_mat,projection_str,symmetric=False,lin_out=True,normalize=True):
    '''
    Compress unlinsoap using u_mat

    unlinsoap.shape = (Nsample,nspecies, nspecies, nmax, nmax, lmax+1)
    u_mat.shape =
    p2.shape = ()
    '''

    # projection
    p = np.einsum(projection_str,u_mat,u_mat,unlinsoap,optimize='optimal')

    p = transform_feature_matrix(p, symmetric=symmetric, normalize=normalize, lin_out=lin_out)

    return p

def transform_feature_matrix(p, symmetric=False, normalize=True, lin_out=True):
    dtype = p.dtype
    Ndim = len(p.shape)

    if Ndim == 6:
        Nsoap,nspecies, nspecies, nmax, nmax, lmax1 = p.shape
        shape1 = (Nsoap,1,1,1,1,1)
        trans = True
    elif Ndim == 4:
        Nsoap,Ncomp , Ncomp , lmax1 = p.shape
        shape1 = (Nsoap,1,1,1)
        trans = False
    elif Ndim == 5:
        Nsoap, nspecies2, nmax, nmax, lmax1 = p.shape
        nspecies = int(np.sqrt(nspecies2))
        p = p.reshape((Nsoap, nspecies, nspecies, nmax, nmax, lmax1))
        shape1 = (Nsoap,1,1,1,1,1)
        trans = True

    if normalize is True:
        pn = np.linalg.norm(p.reshape((Nsoap,-1)),axis=1).reshape(shape1)
        p /=  pn

    if symmetric is True:
        p = p.transpose(0,1,3,2,4,5).reshape(Nsoap,nspecies*nmax, nspecies*nmax, lmax1) if trans is True else p
        p = symmetrize(p,dtype)

    if lin_out:
        return p.reshape((Nsoap,-1))
    else:
        return p
