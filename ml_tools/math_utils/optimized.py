from numba import njit,prange,void,float64,float32,int32,int64,vectorize,jit,config,threading_layer
import math
from ..base import np
from autograd.extend import primitive,defvjp,defjvp
from autograd.numpy.numpy_vjps import unbroadcast_f,replace_zero
from numpy import zeros
@primitive
@vectorize(["float64(float64, float64)","float64(float64, int32)",
            "float32(float32, float64)","float32(float32, int32)"],target='parallel')
def power(x,zeta):
    return math.pow(x,zeta)

defvjp(power,
    lambda ans, x, y : unbroadcast_f(x, lambda g: g * y * power(x,np.where(y, y - 1, 1.))),
    lambda ans, x, y : unbroadcast_f(y, lambda g: g * np.log(replace_zero(x, 1.)) * power(x,y))
      )

defjvp(power,
    lambda g, ans, x, y : g * y * power(x,y-1),
    lambda g, ans, x, y : g * np.log(replace_zero(x, 1.)) * power(x,y)
      )


# @primitive
# def average_kernel(envKernel,Xstrides,Ystrides,is_square):
#     N,M = len(Xstrides)-1,len(Ystrides)-1

#     if is_square is False:
#         ids = np.zeros((N*M,6),np.int32)
#     elif is_square is True:
#         ids = np.zeros((N*(N+1)/2,6),np.int32)
#     print('a')
#     iii = 0
#     for ii,(ist,ind) in enumerate(zip(Xstrides[:-1],Xstrides[1:])):
#         for jj,(jst,jnd) in enumerate(zip(Ystrides[:-1],Ystrides[1:])):
#             if is_square is True:
#                 # computes only lower triangular
#                 if ii < jj: continue
#             ids[iii,:] = np.asarray([ii,ist,ind,jj,jst,jnd],np.int32)
#             iii += 1
#     print('b')
#     K = np.zeros((N,M),order='C')
#     get_average(K,envKernel,ids)
#     print('c')
#     if is_square is True:
#         K += np.tril(K,k=-1).T

#     return K

# @njit([void(float64[:,:], float64[:,:],int32[:,:]),
#         void(float32[:,:], float32[:,:],int32[:,:])],parallel=True)
# def get_average(kernel,env_kernel,ids):
#     for it in prange(len(ids)):
#         ii,sla_st,sla_nd,jj,slb_st,slb_nd = ids[it,:]
#         for jt in prange(sla_st,sla_nd):
#             for kt in prange(slb_st,slb_nd):
#                 kernel[ii,jj] += env_kernel[jt,kt]
#         kernel[ii,jj] /= (sla_nd-sla_st)*(slb_nd-slb_st)

@primitive
def average_kernel(envKernel,Xstrides,Ystrides,is_square):
    N,M = len(Xstrides)-1,len(Ystrides)-1
    K = np.zeros((N,M),order='C')
    ids_n = np.asarray(Xstrides,dtype=np.int32)

    if is_square is False:
        ids_m = np.asarray(Ystrides,dtype=np.int32)
        get_average_rectangular(K,envKernel,ids_n,ids_m)

    elif is_square is True:
        # computes only lower triangular
        get_average_square(K,envKernel,ids_n)
        K += np.tril(K,k=-1).T

    return K

@njit([void(float64[:,:], float64[:,:],int32[:]),
        void(float32[:,:], float32[:,:],int32[:])],parallel=True)
def get_average_square(kernel,env_kernel,ids_n):
    Nenv = ids_n.shape[0]-1
    for it in prange(Nenv):
        ist,ind = ids_n[it],ids_n[it+1]
        for jt in prange(Nenv):
            if it < jt: continue
            jst,jnd = ids_n[jt],ids_n[jt+1]
            for ii in prange(ist,ind):
                # computes only lower triangular
                for jj in prange(jst,jnd):
                    kernel[it,jt] += env_kernel[ii,jj]
            kernel[it,jt] /= (ind-ist)*(jnd-jst)


@njit([void(float64[:,:], float64[:,:],int32[:],int32[:]),
        void(float32[:,:], float32[:,:],int32[:],int32[:])],parallel=True)
def get_average_rectangular(kernel,env_kernel,ids_n,ids_m):
    Nenv = ids_n.shape[0]-1
    Menv = ids_m.shape[0]-1
    for it in prange(Nenv):
        ist,ind = ids_n[it],ids_n[it+1]
        for jt in prange(Menv):
            jst,jnd = ids_m[jt],ids_m[jt+1]
            for ii in prange(ist,ind):
                # computes only lower triangular
                for jj in prange(jst,jnd):
                    kernel[it,jt] += env_kernel[ii,jj]
            kernel[it,jt] /= (ind-ist)*(jnd-jst)

# TODO see how to make the gradients on chunks too
def grad_average_kernel(ans, envKernel,Xstrides,Ystrides,is_square):
    shape = list(envKernel.shape)
    new_shape = [len(Xstrides)-1,len(Ystrides)-1]
    def vjp(g):
        g_repeated = np.zeros(shape)
        N,M = new_shape
        ids = np.zeros((N*M,6),np.int32)
        for ii,(ist,ind) in enumerate(zip(Xstrides[:-1],Xstrides[1:])):
            for jj,(jst,jnd) in enumerate(zip(Ystrides[:-1],Ystrides[1:])):
                ids[ii*M+jj,:] = np.asarray([ii,ist,ind,jj,jst,jnd],np.int32)

        grad_average_kernel_helper(g_repeated,g,ids)
        return g_repeated
    return vjp

@njit([void(float64[:,:], float64[:,:],int32[:,:]),
        void(float32[:,:], float32[:,:],int32[:,:])],parallel=True)
def grad_average_kernel_helper(g_repeated,g,ids):
    for it in prange(len(ids)):
        I,sla_st,sla_nd,J,slb_st,slb_nd = ids[it,:]
        for jt in prange(sla_st,sla_nd):
            aa = (sla_nd-sla_st)
            for kt in range(slb_st,slb_nd):
                g_repeated[jt,kt] = g[I,J]/(aa*(slb_nd-slb_st))

defvjp(average_kernel,grad_average_kernel,None,None)



def symmetrize(p,dtype=np.float64):
    Nsoap,Ncomp,_,nn = p.shape
    p2 = np.zeros((Nsoap,Ncomp*(Ncomp + 1)/2, nn),dtype=dtype)
    stride = [0] + list(range(Ncomp,0,-1))
    stride = np.cumsum(stride)
    ids = []
    for i,st,nd in zip(range(Ncomp-1),stride[:-1],stride[1:]):
        ids.append([i,st,nd])
    ids = np.asarray(ids,dtype=np.int32).reshape((-1,3))
    symm_0(p,p2,ids,Nsoap,Ncomp)
    return p2

def get_unlin_soap(rawsoap,params,global_species,dtype=np.float64):
    """
    Take soap vector from QUIP and undo the vectorization
    (nspecies is symmetric and sometimes nmax so some terms don't exist in the soap vector of QUIP)
    """
    nmax = params['nmax']
    lmax = params['lmax']
    #nn = nmax**2*(lmax + 1)
    nspecies = len(global_species)
    Nsoap = rawsoap.shape[0]

    #translate the fingerprints from QUIP
    p = np.zeros((Nsoap,nspecies, nspecies, nmax, nmax, lmax + 1),dtype=dtype)
    rs_index = np.asarray([(i%nmax, (i - i%nmax)/nmax) for i in xrange(nmax*nspecies)],dtype=np.int32)
    counter = 0
    fac = np.zeros((nmax*nspecies,nmax*nspecies,2),dtype=np.int32)
    for i in xrange(nmax*nspecies):
        for j in xrange(i + 1):
            mult = 0  if i != j else 1

            fac[i,j] = np.array([mult,counter])
            counter += (lmax + 1)

    ff = np.array([np.sqrt(0.5),1.0])
    Nframe = rawsoap.shape[0]
    opt_0(rawsoap,p,rs_index,fac,ff,Nframe,nmax,nspecies,lmax)

    return p


@njit(parallel=True)
def opt_0(rawsoap,p,rs_index,fac,ff,Nframe,nmax,nspecies,lmax):
    for iframe in prange(Nframe):
        for i in prange(nmax*nspecies):
            n1, s1 = rs_index[i,0],rs_index[i,1]
            for j in prange(i + 1):
                mult = ff[fac[i,j,0]]
                counter = fac[i,j,1]
                n2, s2 = rs_index[j,0],rs_index[j,1]

                p[iframe,s1, s2, n1, n2, :] = rawsoap[iframe,counter:counter+(lmax + 1)]*mult
                if s1 == s2: p[iframe,s1, s2, n2, n1, :] = rawsoap[iframe,counter:counter+(lmax + 1)]*mult

    for iframe in prange(Nframe):
        for s1 in prange(nspecies):
            for s2 in prange(s1):
                p[iframe,s2, s1] = p[iframe,s1, s2].transpose((1,0,2))

@njit([void(float64[:,:,:,:], float64[:,:,:],int32[:,:],int32,int32),
        void(float32[:,:,:,:], float32[:,:,:],int32[:,:],int32,int32),
        void(float64[:,:,:,:], float64[:,:,:],int32[:,:],int64,int64),
        void(float32[:,:,:,:], float32[:,:,:],int32[:,:],int64,int64)],parallel=True)
def symm_0(p,p2,ids,Nsoap,Ncomp):
    N = len(ids)
    fac = math.sqrt(2.0)
    for iframe in prange(Nsoap):
        for ii in prange(N):
            i,st,nd = ids[ii,0],ids[ii,1],ids[ii,2]
            p2[iframe,st] = p[iframe,i, i]
            p2[iframe,st+1:nd] = p[iframe,i, (i+1):Ncomp]*fac
        p2[iframe,-1] = p[iframe,Ncomp-1,Ncomp-1]