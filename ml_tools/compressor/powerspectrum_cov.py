from ..base import np,sp
from ..base import BaseEstimator,TransformerMixin

class CompressorCovarianceUmat(BaseEstimator,TransformerMixin):
    def __init__(self,soap_params=None,compression_type='species',dj=5,fj=None,symmetric=False,to_reshape=True):
        self.dj = dj
        self.compression_type = compression_type
        self.soap_params = soap_params
        self.fj = fj
        self.symmetric = symmetric
        self.to_reshape = to_reshape
    def get_params(self,deep=True):
        params = dict(dj=self.dj,compression_type=self.compression_type,
                      soap_params=self.soap_params,fj=self.fj,
                      symmetric=self.symmetric)
        return params
    
    def set_params(self,params):
        self.dj = params['dj']
        self.compression_type = params['compression_type']
        self.soap_params = params['soap_params']
        self.fj = params['fj']
        self.symmetric = params['symmetric']
    def set_fj(self,fj):
        #self.u_mat = None
        self.fj = fj
        self.dj = len(fj)
    def reshape_(self,X):
        unwrapped_X = get_unlin_soap(X,self.soap_params,self.soap_params['global_species'])
        
        Nsample,nspecies, nspecies, nmax, nmax, lmax1 =  unwrapped_X.shape
        
        if self.compression_type == 'species':
            X_c = unwrapped_X.reshape((Nsample,nspecies, nspecies,nmax**2*lmax1))
        elif self.compression_type == 'radial':
            X_c = unwrapped_X.transpose(0,3,4,1,2,5).reshape((Nsample,nmax, nmax,nspecies**2*lmax1))
        elif self.compression_type == 'species+radial':
            X_c = unwrapped_X.transpose(0,1,3,2,4,5).reshape((Nsample,nspecies*nmax, nspecies*nmax,lmax1))
        return X_c
    
    def fit(self,X):
        
        if self.to_reshape:
            X_c = self.reshape_(X)
        else:
            X_c = X
        
        self.u_mat_full, self.eig = get_covariance_umat_full(X_c)
        
        return self
    
    def transform(self,X):
        if self.to_reshape:
            X_c = self.reshape_(X)
        else:
            X_c = X
        
        if self.fj is not None:
            #u_mat = np.dot(self.fj,self.u_mat_full[:self.dj,:])
            u_mat = np.einsum("ja,j->ja",self.u_mat_full[:self.dj,:],self.fj,optimize='optimal')
        else:
            u_mat = self.u_mat_full[:self.dj,:]
        return get_compressed_soap(X_c,u_mat,symmetric=self.symmetric)
    
    def fit_transform(self,X):
        if self.to_reshape:
            X_c = self.reshape_(X)
        else:
            X_c = X

        self.u_mat_full,self.eig = get_covariance_umat_full(X_c)
        if self.fj is not None:
            u_mat = np.dot(self.fj,self.u_mat_full[:self.dj,:])
            #u_mat = np.einsum("ja,j->ja",self.u_mat_full[:self.dj,:],self.fj,optimize='optimal')
        else:
            u_mat = self.u_mat_full[:self.dj,:]
        
        return get_compressed_soap(X_c,u_mat,symmetric=self.symmetric)

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

    
def get_covariance_umat_full(unlinsoap):
    '''
    Compute the covariance of the given unlinsoap and decomposes it 
    unlinsoap.shape = (Nsoap,Nfull,Nfull,nn)
    '''
    cov = unlinsoap.mean(axis=(0,3))
    eva,eve = np.linalg.eigh(cov)
    eva = np.flip(eva,axis=0)
    eve = np.flip(eve,axis=1)
    #eve = eve*np.sqrt(eva).reshape((1,-1))

    return eve.T,eva.flatten()


def get_compressed_soap(unlinsoap,u_mat,symmetric=False,lin_out=True):
    '''
    Compress unlinsoap dim 1 and 2 using u_mat

    unlinsoap.shape = (Nsoap,Nfull,Nfull,nn)
    u_mat.shape = (Ncomp,Nfull)
    p2.shape = (Nsoap,Ncomp*(Ncomp+1)/2,nn)
    '''
    Nsoap,_,_,nn = unlinsoap.shape
    
    Ncomp = u_mat.shape[0]
    # projection 
    #p = np.zeros((Nsoap,Ncomp,Ncomp,nn))
    p = np.einsum('ij,ek,ajkm->aiem',u_mat,u_mat,unlinsoap,optimize='optimal')
    pn = np.linalg.norm(p.reshape((Nsoap,-1)),axis=1).reshape((Nsoap,1,1,1))
    p /=  pn
    if symmetric is True:
        p2 = np.empty((Nsoap,Ncomp*(Ncomp + 1)/2, nn))
        stride = [0] + list(range(Ncomp,0,-1))
        stride = np.cumsum(stride)
        for i,st,nd in zip(range(Ncomp-1),stride[:-1],stride[1:]):
            p2[:,st] = p[:,i, i]
            p2[:,st+1:nd] = p[:,i, (i+1):Ncomp]*np.sqrt(2.0)
        p2[:,-1] = p[:,Ncomp-1,Ncomp-1]
        if lin_out:
            return p2.reshape((Nsoap,Ncomp*(Ncomp + 1)/2 * nn))
        else:
            return p2
    else:
        if lin_out:
            return p.reshape((Nsoap,Ncomp**2 * nn))
        else:
            return p

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
