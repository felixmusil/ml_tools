from ..base import np,sp
from ..base import BaseEstimator,TransformerMixin
from scipy.sparse import lil_matrix,csr_matrix,issparse
from ..utils import tqdm_cs,s2hms
from time import time
from ..math_utils import symmetrize,get_unlin_soap
  
class CompressorCovarianceUmat(BaseEstimator,TransformerMixin):
    def __init__(self,soap_params=None,compression_type='species',dj=5,fj=None,stride_size=None,
                    symmetric=False,to_reshape=True,normalize=True,dtype='float64',full_opt=False):
        self.dj = dj
        self.compression_type = compression_type
        self.soap_params = soap_params
        self.fj = fj
        self.symmetric = symmetric
        self.to_reshape = to_reshape
        self.normalize = normalize
        self.dtype = dtype
        self.stride_size = stride_size
        self.full_opt = full_opt

    def get_params(self,deep=True):
        params = dict(dj=self.dj,compression_type=self.compression_type,
                      soap_params=self.soap_params,fj=self.fj,
                      dtype=self.dtype,stride_size=self.stride_size,
                      scale_features_str=self.scale_features_str,
                      full_opt=self.full_opt,
                      symmetric=self.symmetric,to_reshape=self.to_reshape,
                      einsum_str=self.einsum_str,normalize=self.normalize)
        return params
    
    def set_params(self,params):
        self.dj = params['dj']
        self.compression_type = params['compression_type']
        self.soap_params = params['soap_params']
        self.fj = params['fj']
        self.dtype = params['dtype']
        self.symmetric = params['symmetric']
        self.to_reshape = params['to_reshape']
        self.einsum_str = params['einsum_str']
        self.normalize = params['normalize']
        self.scale_features_str = params['scale_features_str']
        self.stride_size = params['stride_size']
        self.full_opt = params['full_opt']

        nspecies = len(self.soap_params['global_species'])
        lmax1 = self.soap_params['lmax'] + 1
        nmax = self.soap_params['nmax']
        identity = lambda x: x 
        reshape = lambda x: x.transpose(0,1,3,2,4,5).reshape((-1,nspecies*nmax, nspecies*nmax,lmax1))
        if 'species*radial' in self.compression_type:
            self.modify = reshape
        else:
            self.modify = identity

    def set_scaling_factors(self,factors):
        if self.full_opt is False:
            self.set_fj(factors)
        elif self.full_opt is True:
            self.set_uj(factors)

    def set_fj(self,x,xl=None):
        #self.u_mat = None
        if 'angular' in self.compression_type and xl is None:
            # angular at the front and rest after
            self.dl = self.soap_params['lmax']+1
            self.fl = x[:self.dl]
            self.fj = x[self.dl:]
        elif 'angular' in self.compression_type and xl is not None:
            self.fl = xl
            self.dl = len(xl)
            self.fj = x
        else:
            self.fj = x
        self.dj = len(self.fj)

    def set_uj(self,u_mat):
        if 'angular' in self.compression_type:
            # self.dj = u_mat.shape[1]
            self.dl = self.soap_params['lmax']+1
        elif 'angular' not in self.compression_type:
            # self.dj = u_mat.shape[0]
            self.dl = 0
        
        self.fj = None
        self.uj = u_mat.reshape((self.dj,self.dj))

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
        
        identity = lambda x: x 
        reshape = lambda x: x.transpose(0,1,3,2,4,5).reshape((-1,nspecies*nmax, nspecies*nmax,lmax1))
        if self.compression_type in ['species']:
            cov = X.mean(axis=(4)).trace(axis1=2, axis2=3)
            self.einsum_str = 'ij,nm,ajmopl->ainopl'
            self.scale_features_str = 'j,m,ajmopl->ajmopl'
            self.modify = identity
        elif self.compression_type in ['angular+species']:
            X_c = X.transpose(4,0,1,2,3)
            cov = X_c.trace(axis1=3, axis2=4)
            self.einsum_str = 'ij,nm,ajmopl->ainopl'
            self.scale_features_str = 'l,j,m,ajmopl->ajmopl'
            self.modify = identity
        elif self.compression_type in ['radial']:
            cov = X.mean(axis=(4)).trace(axis1=0, axis2=1)
            self.einsum_str = 'ij,nm,aopjml->aopinl'
            self.scale_features_str = 'j,m,aopjml->aopjml'
            self.modify = identity
        elif self.compression_type in ['angular+radial']:
            X_c = X.transpose(4,0,1,2,3)
            cov = X_c.trace(axis1=1, axis2=2)
            self.einsum_str = 'ij,nm,aopjml->aopinl'
            self.scale_features_str = 'l,j,m,aopjml->aopjml'
            self.modify = identity
        elif self.compression_type in ['species*radial']:
            X_c = X.transpose(0,2,1,3,4).reshape((nspecies*nmax, nspecies*nmax,lmax1))
            cov = X_c.mean(axis=(2))
            
            self.einsum_str = 'ij,nm,ajml->ainl'
            self.scale_features_str = 'j,m,ajml->ajml'
            # there is a contraction here so we need to reshape the input before transforming it
            self.modify = reshape
        elif self.compression_type in ['angular+species*radial']:
            X_c = X.transpose(4,0,2,1,3).reshape((lmax1,nspecies*nmax, nspecies*nmax))
            cov = X_c
            self.einsum_str = 'ij,nm,ajml->ainl'
            self.scale_features_str = 'l,j,m,ajml->ajml'
            self.modify = reshape

        self.scale_features_full_str = self.einsum_str
        # get sorted eigen values and vectors
        eva,eve = np.linalg.eigh(cov)
        # first dim is angular so the flip has to be shifted
        if 'angular' not in self.compression_type:
            eva = np.flip(eva,axis=0)
            eve = np.flip(eve,axis=1)
            return eve.T,eva.flatten()
        elif 'angular' in self.compression_type:
            self.einsum_str = self.einsum_str.replace('ij','lij').replace('nm','lnm')
            eva = np.flip(eva,axis=1)
            eve = np.flip(eve,axis=2)
            return eve.transpose((0,2,1)),eva
        
    def fit(self,X):
        if issparse(X) is True:
            X_c = X
        elif self.to_reshape is True:
            X_c = self.reshape_(np.asarray(X,dtype=self.dtype))
        else:
            X_c = np.asarray(X,dtype=self.dtype)
        
        self.u_mat_full, self.eig = self.get_covariance_umat_full(X_c)
        
        return self
    
    def project_on_u_mat(self,X,compression_only=False,stride_size=None,fj_mult=False):

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

            if fj_mult is True:
                if self.fj is not None and 'angular' not in self.compression_type:
                    #u_mat = np.dot(self.fj,self.u_mat_full[:self.dj,:])
                    u_mat = np.einsum("ja,j->ja",self.u_mat_full[:self.dj,:],self.fj,optimize='optimal')
                elif self.fj is not None and 'angular' in self.compression_type:
                    #u_mat = np.dot(self.fj,self.u_mat_full[:self.dj,:])
                    u_mat = np.einsum("lnm,n->lnm",self.u_mat_full[:,:self.dj,:],self.fj,optimize='optimal')
                    u_mat = np.einsum("lnm,l->lnm",u_mat,self.fl,optimize='optimal')
                elif self.fj is None and 'angular' not in self.compression_type:
                    u_mat = self.u_mat_full[:self.dj,:]
                elif self.fj is None and 'angular' in self.compression_type:
                    u_mat = self.u_mat_full[:,:self.dj,:]
            else:
                if 'angular' not in self.compression_type:
                    u_mat = np.array(self.u_mat_full[:self.dj,:],dtype=self.dtype)
                elif 'angular' in self.compression_type:
                    u_mat = np.array(self.u_mat_full[:,:self.dj,:],dtype=self.dtype)
                
            if compression_only is False:
                X_compressed.append(get_compressed_soap(X_c,u_mat,self.einsum_str,symmetric=False,
                                    lin_out=False,normalize=self.normalize))
            elif compression_only is True:
                X_compressed.append(get_compressed_soap(X_c,u_mat,self.einsum_str,symmetric=True,
                                    lin_out=True,normalize=self.normalize))

        if issparse(X) is False and stride_size is None:
            return X_compressed[0]
        elif issparse(X) is True or stride_size is not None:
            aa = np.array(np.concatenate(X_compressed,axis=0),dtype=self.dtype)
            return aa

    def scale_features(self,projected_unlinsoap,stride_size=None):
        
        if self.full_opt is False:
            return self.scale_features_diag(projected_unlinsoap,stride_size)
        elif self.full_opt is True:
            return self.scale_features_full(projected_unlinsoap,stride_size)

    def get_bounds(self,N,stride_size=None):
        if stride_size is None:
            bounds = [(0,N)] 
        elif stride_size is not None:
            Nstride = N // stride_size
            bounds = [[ii*stride_size,(ii+1)*stride_size] for ii in range(Nstride)]
            bounds[-1][-1] += N % stride_size
        return bounds

    def scale_features_diag(self,projected_unlinsoap,stride_size=None):
        N = projected_unlinsoap.shape[0]
        X_compressed = []
        bounds = self.get_bounds(N,stride_size)
        for st,nd in tqdm_cs(bounds,desc='Feature Scaling Diag',leave=False):
            args = [self.scale_features_str]
            X_c = projected_unlinsoap[st:nd]
            if 'angular' not in self.compression_type:
                args += [self.fj,self.fj,X_c]
            elif 'angular' in self.compression_type:
                args += [self.fl,self.fj,self.fj,X_c]
            
            kwargs = dict(optimize=False)
            p = np.einsum(*args,**kwargs)
            if len(p.shape) == 6:
                Nsoap,nspecies, nspecies, nmax, nmax, lmax1 = p.shape
                shape1 = (Nsoap,1,1,1,1,1)
                axis = (1,2,3,4,5)
                trans = True
            elif len(p.shape) == 4:
                Nsoap,Ncomp , Ncomp , lmax1 = p.shape
                shape1 = (Nsoap,1,1,1)
                axis = (1,2,3)
                trans = False

            if self.symmetric is True:
                p = np.transpose(p,axes=(0,1,3,2,4,5)).reshape(
                    Nsoap,nspecies*nmax, nspecies*nmax, lmax1) if trans is True else p
                p = symmetrize(p)
                shape1 = (Nsoap,1,1)

            if self.normalize is True:
                pn = np.sqrt(np.sum(np.square(p),axis=axis)).reshape(shape1)
                p /=  pn

            X_compressed.append(p.reshape((Nsoap,-1)))

        if stride_size is None:
            return X_compressed[0]
        elif stride_size is not None:
            aa = np.concatenate(X_compressed,axis=0)
            return aa

    def scale_features_full(self,projected_unlinsoap,stride_size=None):
        N = projected_unlinsoap.shape[0]
        bounds = self.get_bounds(N,stride_size)
        X_compressed = []
        
        for st,nd in tqdm_cs(bounds,desc='Feature Scaling Full',leave=False):
            args = [self.scale_features_full_str]
            X_c = projected_unlinsoap[st:nd]
            if 'angular' not in self.compression_type:
                args += [self.uj,self.uj,X_c]
            elif 'angular' in self.compression_type:
                raise Exception('Not implemented')
            
            kwargs = dict(optimize=False)
            p = np.einsum(*args,**kwargs)
            if len(p.shape) == 6:
                Nsoap,nspecies, nspecies, nmax, nmax, lmax1 = p.shape
                shape1 = (Nsoap,1,1,1,1,1)
                axis = (1,2,3,4,5)
                trans = True
            elif len(p.shape) == 4:
                Nsoap,Ncomp , Ncomp , lmax1 = p.shape
                shape1 = (Nsoap,1,1,1)
                axis = (1,2,3)
                trans = False

            if self.symmetric is True:
                p = np.transpose(p,axes=(0,1,3,2,4,5)).reshape(Nsoap,nspecies*nmax, nspecies*nmax, lmax1) if trans is True else p
                p = symmetrize(p)
                shape1 = (Nsoap,1,1)

            if self.normalize is True:
                pn = np.sqrt(np.sum(np.square(p),axis=axis)).reshape(shape1)
                p /=  pn

            X_compressed.append(p.reshape((Nsoap,-1)))

        if stride_size is None:
            return X_compressed[0]
        elif stride_size is not None:
            aa = np.concatenate(X_compressed,axis=0)
            return aa
        
    def transform(self,X):
        X_proj = self.project_on_u_mat(X,compression_only=True,stride_size=self.stride_size,
                                fj_mult=True)
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
        
    
def get_compressed_soap(unlinsoap,u_mat,einsum_str,symmetric=False,lin_out=True,normalize=True):
    '''
    Compress unlinsoap using u_mat

    unlinsoap.shape = (Nsample,nspecies, nspecies, nmax, nmax, lmax+1)
    u_mat.shape = 
    p2.shape = ()
    '''
    
    # projection 
    p = np.einsum(einsum_str,u_mat,u_mat,unlinsoap,optimize='optimal')
    dtype = p.dtype
    if len(unlinsoap.shape) == 6:
        Nsoap,nspecies, nspecies, nmax, nmax, lmax1 = p.shape
        shape1 = (Nsoap,1,1,1,1,1)
        trans = True
    elif len(unlinsoap.shape) == 4:
        Nsoap,Ncomp , Ncomp , lmax1 = p.shape
        shape1 = (Nsoap,1,1,1)
        trans = False

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


    

    



# class CompressorCovarianceUmat_old(BaseEstimator,TransformerMixin):
#     def __init__(self,soap_params=None,compression_type='species',dj=5,fj=None,symmetric=False,to_reshape=True):
#         self.dj = dj
#         self.compression_type = compression_type
#         self.soap_params = soap_params
#         self.fj = fj
#         self.symmetric = symmetric
#         self.to_reshape = to_reshape
#     def get_params(self,deep=True):
#         params = dict(dj=self.dj,compression_type=self.compression_type,
#                       soap_params=self.soap_params,fj=self.fj,
#                       symmetric=self.symmetric,to_reshape=self.to_reshape)
#         return params
    
#     def set_params(self,params):
#         self.dj = params['dj']
#         self.compression_type = params['compression_type']
#         self.soap_params = params['soap_params']
#         self.fj = params['fj']
#         self.symmetric = params['symmetric']
#         self.to_reshape = params['to_reshape']

        
#     def set_fj(self,fj):
#         #self.u_mat = None
#         self.fj = fj
#         self.dj = len(fj)
#     def reshape_(self,X):
#         unwrapped_X = get_unlin_soap(X,self.soap_params,self.soap_params['global_species'])
        
#         Nsample,nspecies, nspecies, nmax, nmax, lmax1 =  unwrapped_X.shape
        
#         if self.compression_type == 'species':
#             X_c = unwrapped_X.reshape((Nsample,nspecies, nspecies,nmax**2*lmax1))
#         elif self.compression_type == 'radial':
#             X_c = unwrapped_X.transpose(0,3,4,1,2,5).reshape((Nsample,nmax, nmax,nspecies**2*lmax1))
#         elif self.compression_type == 'species+radial':
#             X_c = unwrapped_X.transpose(0,1,3,2,4,5).reshape((Nsample,nspecies*nmax, nspecies*nmax,lmax1))
#         return X_c
    
#     def fit(self,X):
        
#         if self.to_reshape:
#             X_c = self.reshape_(X)
#         else:
#             X_c = X
        
#         self.u_mat_full, self.eig = get_covariance_umat_full(X_c)
        
#         return self
    
#     def transform(self,X):
#         if self.to_reshape:
#             X_c = self.reshape_(X)
#         else:
#             X_c = X
        
#         if self.fj is not None:
#             #u_mat = np.dot(self.fj,self.u_mat_full[:self.dj,:])
#             u_mat = np.einsum("ja,j->ja",self.u_mat_full[:self.dj,:],self.fj,optimize='optimal')
#         else:
#             u_mat = self.u_mat_full[:self.dj,:]
#         return get_compressed_soap(X_c,u_mat,symmetric=self.symmetric)
    
#     def fit_transform(self,X):
#         if self.to_reshape:
#             X_c = self.reshape_(X)
#         else:
#             X_c = X

#         self.u_mat_full,self.eig = get_covariance_umat_full(X_c)
#         if self.fj is not None:
#             u_mat = np.dot(self.fj,self.u_mat_full[:self.dj,:])
#             #u_mat = np.einsum("ja,j->ja",self.u_mat_full[:self.dj,:],self.fj,optimize='optimal')
#         else:
#             u_mat = self.u_mat_full[:self.dj,:]
        
#         return get_compressed_soap(X_c,u_mat,symmetric=self.symmetric)

#     def pack(self):
#         params = self.get_params()
#         data = dict(u_mat_full=self.u_mat_full.tolist(),
#                     eig=self.eig.tolist())
#         state = dict(data=data,
#                      params=params)
#         return state

#     def unpack(self,state):
#         self.set_params(state['params'])
#         self.u_mat_full = np.array(state['data']['u_mat_full'])
#         self.eig = np.array(state['data']['eig'])


# def get_covariance_umat_full_old(unlinsoap):
#     '''
#     Compute the covariance of the given unlinsoap and decomposes it 
#     unlinsoap.shape = (Nsoap,Nfull,Nfull,nn)
#     '''
#     cov = unlinsoap.mean(axis=(0,3))
#     eva,eve = np.linalg.eigh(cov)
#     eva = np.flip(eva,axis=0)
#     eve = np.flip(eve,axis=1)
#     #eve = eve*np.sqrt(eva).reshape((1,-1))

#     return eve.T,eva.flatten()