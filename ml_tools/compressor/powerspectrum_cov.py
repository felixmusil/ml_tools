from ..base import np,sp
from ..base import BaseEstimator,TransformerMixin
 
class CompressorCovarianceUmat(BaseEstimator,TransformerMixin):
    def __init__(self,soap_params=None,compression_type='species',dj=5,fj=None,
                    symmetric=False,to_reshape=True,normalize=True):
        self.dj = dj
        self.compression_type = compression_type
        self.soap_params = soap_params
        self.fj = fj
        self.symmetric = symmetric
        self.to_reshape = to_reshape
        self.normalize = normalize

    def get_params(self,deep=True):
        params = dict(dj=self.dj,compression_type=self.compression_type,
                      soap_params=self.soap_params,fj=self.fj,
                      scale_features_str=self.scale_features_str,
                      symmetric=self.symmetric,to_reshape=self.to_reshape,
                      einsum_str=self.einsum_str,normalize=self.normalize)
        return params
    
    def set_params(self,params):
        self.dj = params['dj']
        self.compression_type = params['compression_type']
        self.soap_params = params['soap_params']
        self.fj = params['fj']
        self.symmetric = params['symmetric']
        self.to_reshape = params['to_reshape']
        self.einsum_str = params['einsum_str']
        self.normalize = params['normalize']
        self.scale_features_str = params['scale_features_str']

        nspecies = len(self.soap_params['global_species'])
        lmax1 = self.soap_params['lmax'] + 1
        nmax = self.soap_params['nmax']
        identity = lambda x: x 
        reshape = lambda x: x.transpose(0,1,3,2,4,5).reshape((-1,nspecies*nmax, nspecies*nmax,lmax1))
        if 'species*radial' in self.compression_type:
            self.modify = reshape
        else:
            self.modify = identity

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

    def reshape_(self,X):
        unwrapped_X = get_unlin_soap(X,self.soap_params,self.soap_params['global_species'])
        return unwrapped_X

    def get_covariance_umat_full(self,unlinsoap):
        '''
        Compute the covariance of the given unlinsoap and decomposes it 
        unlinsoap.shape = (Nsample,nspecies, nspecies, nmax, nmax, lmax+1)
        '''
        Nsample,nspecies, nspecies, nmax, nmax, lmax1 =  unlinsoap.shape
        
        identity = lambda x: x 
        reshape = lambda x: x.transpose(0,1,3,2,4,5).reshape((-1,nspecies*nmax, nspecies*nmax,lmax1))
        if self.compression_type in ['species']:
            cov = unlinsoap.mean(axis=(0,5)).trace(axis1=2, axis2=3)
            self.einsum_str = 'ij,nm,ajmopl->ainopl'
            self.scale_features_str = 'j,m,ajmopl->ajmopl'
            self.modify = identity
        elif self.compression_type in ['angular+species']:
            X_c = unlinsoap.transpose(5,0,1,2,3,4)
            cov = X_c.mean(axis=1).trace(axis1=3, axis2=4)
            self.einsum_str = 'ij,nm,ajmopl->ainopl'
            self.scale_features_str = 'l,j,m,ajmopl->ajmopl'
            self.modify = identity
        elif self.compression_type in ['radial']:
            cov = unlinsoap.mean(axis=(0,5)).trace(axis1=0, axis2=1)
            self.einsum_str = 'ij,nm,aopjml->aopinl'
            self.scale_features_str = 'j,m,aopjml->aopjml'
            self.modify = identity
        elif self.compression_type in ['angular+radial']:
            X_c = unlinsoap.transpose(5,0,1,2,3,4)
            cov = X_c.mean(axis=1).trace(axis1=1, axis2=2)
            self.einsum_str = 'ij,nm,aopjml->aopinl'
            self.scale_features_str = 'l,j,m,aopjml->aopjml'
            self.modify = identity
        elif self.compression_type in ['species*radial']:
            X_c = unlinsoap.transpose(0,1,3,2,4,5).reshape((Nsample,nspecies*nmax, nspecies*nmax,lmax1))
            cov = X_c.mean(axis=(0,3))
            
            self.einsum_str = 'ij,nm,ajml->ainl'
            self.scale_features_str = 'j,m,ajml->ajml'
            # there is a contraction here so we need to reshape the input before transforming it
            self.modify = reshape
        elif self.compression_type in ['angular+species*radial']:
            X_c = unlinsoap.transpose(5,0,1,3,2,4).reshape((lmax1,Nsample,nspecies*nmax, nspecies*nmax))
            cov = X_c.mean(axis=1)
            self.einsum_str = 'ij,nm,ajml->ainl'
            self.scale_features_str = 'l,j,m,ajml->ajml'
            self.modify = reshape

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
        
        if self.to_reshape:
            X_c = self.reshape_(X)
        else:
            X_c = X
        
        self.u_mat_full, self.eig = self.get_covariance_umat_full(X_c)
        
        return self
    
    def project_on_u_mat(self,X,compression_only=False):
        if self.to_reshape:
            X_c = self.modify(self.reshape_(X))
        else:
            X_c = self.modify(X)

        if 'angular' not in self.compression_type:
            u_mat = self.u_mat_full[:self.dj,:]
        elif 'angular' in self.compression_type:
            u_mat = self.u_mat_full[:,:self.dj,:]
            
        if compression_only is False:
            return get_compressed_soap(X_c,u_mat,self.einsum_str,symmetric=False,
                                lin_out=False,normalize=self.normalize)
        elif compression_only is True:
            return get_compressed_soap(X_c,u_mat,self.einsum_str,symmetric=True,
                                lin_out=True,normalize=self.normalize)

    def scale_features(self,projected_unlinsoap):
        
        args = [self.scale_features_str]
        if 'angular' not in self.compression_type:
            args += [self.fj,self.fj,projected_unlinsoap]
        elif 'angular' in self.compression_type:
            args += [self.fl,self.fj,self.fj,projected_unlinsoap]
        
        print self.scale_features_str
        print self.fj.shape,projected_unlinsoap.shape
        kwargs = dict(optimize='optimal')
        p = np.einsum(*args,**kwargs)

        if len(p.shape) == 6:
            Nsoap,nspecies, nspecies, nmax, nmax, lmax1 = p.shape
            shape1 = (Nsoap,1,1,1,1,1)
            trans = True
        elif len(p.shape) == 4:
            Nsoap,Ncomp , Ncomp , lmax1 = p.shape
            shape1 = (Nsoap,1,1,1)
            trans = False

        if self.symmetric is True:
            p = np.transpose(p,axes=(0,1,3,2,4,5)).reshape(Nsoap,nspecies*nmax, nspecies*nmax, lmax1) if trans is True else p
            p = symmetrize(p)
            shape1 = (Nsoap,1,1)

        if self.normalize is True:
            pn = np.linalg.norm(p.reshape((Nsoap,-1)),axis=1).reshape(shape1)
            p /=  pn

        return p.reshape((Nsoap,-1))
        
    def transform(self,X):
        if self.to_reshape:
            X_c = self.modify(self.reshape_(X))
        else:
            X_c = self.modify(X)
        
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
        
        
        return get_compressed_soap(X_c,u_mat,self.einsum_str,symmetric=self.symmetric,
                                    normalize=self.normalize)
    
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
        p = symmetrize(p)

    if lin_out:
        return p.reshape((Nsoap,-1))
    else:
        return p
   
def symmetrize(p):
    Nsoap,Ncomp,_,nn = p.shape
    p2 = np.empty((Nsoap,Ncomp*(Ncomp + 1)/2, nn))
    stride = [0] + list(range(Ncomp,0,-1))
    stride = np.cumsum(stride)
    for i,st,nd in zip(range(Ncomp-1),stride[:-1],stride[1:]):
        p2[:,st] = p[:,i, i]
        p2[:,st+1:nd] = p[:,i, (i+1):Ncomp]*np.sqrt(2.0)
    p2[:,-1] = p[:,Ncomp-1,Ncomp-1]
    return p2

def get_unlin_soap(rawsoap,params,global_species):
    """
    Take soap vector from QUIP and undo the vectorization 
    (nspecies is symmetric and sometimes nmax so some terms don't exist in the soap vector of QUIP)
    """
    nmax = params['nmax']
    lmax = params['lmax']
    #nn = nmax**2*(lmax + 1)
    nspecies = len(global_species)
    Nsoap = rawsoap.shape[0]
    #p = np.zeros((nspecies, nspecies, nn))

    #translate the fingerprints from QUIP
    counter = 0
    p = np.zeros((Nsoap,nspecies, nspecies, nmax, nmax, lmax + 1))
    rs_index = [(i%nmax, (i - i%nmax)/nmax) for i in xrange(nmax*nspecies)]
    for i in xrange(nmax*nspecies):
        for j in xrange(i + 1):
            if i != j: mult = np.sqrt(0.5)
            else: mult = 1.0
            n1, s1 = rs_index[i]
            n2, s2 = rs_index[j]
            p[:,s1, s2, n1, n2, :] = rawsoap[:,counter:counter+(lmax + 1)]*mult
            if s1 == s2: p[:,s1, s2, n2, n1, :] = rawsoap[:,counter:counter+(lmax + 1)]*mult
            counter += (lmax + 1)

    for s1 in xrange(nspecies):
        for s2 in xrange(s1):
            p[:,s2, s1] = p[:,s1, s2].transpose((0, 2, 1, 3))
    
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