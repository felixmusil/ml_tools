from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
from ..base import np,sp
from autograd.extend import primitive,defvjp,defjvp

def power(x,zeta):
    return np.power(x,zeta)


def sum_kernel(envKernel,Xstrides,Ystrides,is_square):
    N = len(Xstrides)-1
    M = len(Ystrides)-1
    K = np.zeros((N,M),order='C')
    for ii,(ist,ind) in enumerate(zip(Xstrides[:-1],Xstrides[1:])):
        for jj,(jst,jnd) in enumerate(zip(Ystrides[:-1],Ystrides[1:])):
            if is_square is True:
                if ii < jj: continue
            K[ii,jj] = np.sum(envKernel[ist:ind,jst:jnd])

    if is_square is True:
        K += np.tril(K,k=-1).T
    return K

@primitive
def average_kernel(envKernel,Xstrides,Ystrides,is_square):
    N = len(Xstrides)-1
    M = len(Ystrides)-1
    K = np.zeros((N,M),order='C')
    for ii,(ist,ind) in enumerate(zip(Xstrides[:-1],Xstrides[1:])):
        for jj,(jst,jnd) in enumerate(zip(Ystrides[:-1],Ystrides[1:])):
            if is_square is True:
                if ii < jj: continue
            K[ii,jj] = np.mean(envKernel[ist:ind,jst:jnd])

    if is_square is True:
        K += np.tril(K,k=-1).T
    return K

def grad_average_kernel(ans, envKernel,Xstrides,Ystrides,is_square):
    shape = list(envKernel.shape)
    new_shape = [len(Xstrides)-1,len(Ystrides)-1]
    def vjp(g):
        g_repeated = np.zeros(shape)
        for I,(ist,ind) in enumerate(zip(Xstrides[:-1],Xstrides[1:])):
            for J,(jst,jnd) in enumerate(zip(Ystrides[:-1],Ystrides[1:])):
                if is_square is True:
                    if I < J:
                        g_repeated[ist:ind,jst:jnd] = g_repeated[jst:jnd,ist:ind].T
                        continue
                g_repeated[ist:ind,jst:jnd] = g[I,J] /((ind-ist)*(jnd-jst))
        return g_repeated
    return vjp

defvjp(average_kernel,grad_average_kernel,None,None)


def symmetrize(p):
    Nsoap,Ncomp,_,nn = p.shape
    p2 = np.empty((Nsoap,old_div(Ncomp*(Ncomp + 1),2), nn))
    stride = [0] + list(range(Ncomp,0,-1))
    stride = np.cumsum(stride)
    for i,st,nd in zip(list(range(Ncomp-1)),stride[:-1],stride[1:]):
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
    rs_index = [(i%nmax, old_div((i - i%nmax),nmax)) for i in range(nmax*nspecies)]
    for i in range(nmax*nspecies):
        for j in range(i + 1):
            if i != j: mult = np.sqrt(0.5)
            else: mult = 1.0
            n1, s1 = rs_index[i]
            n2, s2 = rs_index[j]
            p[:,s1, s2, n1, n2, :] = rawsoap[:,counter:counter+(lmax + 1)]*mult
            if s1 == s2: p[:,s1, s2, n2, n1, :] = rawsoap[:,counter:counter+(lmax + 1)]*mult
            counter += (lmax + 1)

    for s1 in range(nspecies):
        for s2 in range(s1):
            p[:,s2, s1] = p[:,s1, s2].transpose((0, 2, 1, 3))

    return p
