from numba import njit,prange,void,float64,int32,vectorize,jit,config,threading_layer
import math
from ..base import np
from autograd.extend import primitive,defvjp,defjvp
from autograd.numpy.numpy_vjps import unbroadcast_f,replace_zero

@primitive
@vectorize(["float64(float64, float64)","float64(float64, int32)"],target='parallel')
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


@primitive
def average_kernel(envKernel,Xstrides,Ystrides):
    N,M = len(Xstrides)-1,len(Ystrides)-1
    ids = np.zeros((N*M,6),np.int32)
    for ii,(ist,ind) in enumerate(zip(Xstrides[:-1],Xstrides[1:])):
        for jj,(jst,jnd) in enumerate(zip(Ystrides[:-1],Ystrides[1:])):
            
            ids[ii*N+jj,:] = np.asarray([ii,ist,ind,jj,jst,jnd],np.int32)
    
    K = np.zeros((N,M),order='C')
    get_average(K,envKernel,ids)
    return K

@njit(void(float64[:,:], float64[:,:],int32[:,:]),parallel=True)
def get_average(kernel,env_kernel,ids):
    for it in prange(len(ids)):
        ii,sla_st,sla_nd,jj,slb_st,slb_nd = ids[it,:]
        for jt in prange(sla_st,sla_nd):
            for kt in prange(slb_st,slb_nd):
                kernel[ii,jj] += env_kernel[jt,kt]
        kernel[ii,jj] /= (sla_nd-sla_st)*(slb_nd-slb_st)


def grad_average_kernel(ans, envKernel,Xstrides,Ystrides):
    shape = list(envKernel.shape)
    new_shape = [len(Xstrides)-1,len(Ystrides)-1]
    def vjp(g):
        g_repeated = np.zeros(shape)
        N,M = new_shape
        ids = np.zeros((N*M,6),np.int32)
        for ii,(ist,ind) in enumerate(zip(Xstrides[:-1],Xstrides[1:])):
            for jj,(jst,jnd) in enumerate(zip(Ystrides[:-1],Ystrides[1:])):
                ids[ii*N+jj,:] = np.asarray([ii,ist,ind,jj,jst,jnd],np.int32)
                
        grad_sum_03_helper(g_repeated,g,ids)
        return g_repeated
    return vjp

@njit(void(float64[:,:], float64[:,:],int32[:,:]),parallel=True)
def grad_sum_03_helper(g_repeated,g,ids):
    for it in prange(len(ids)):
        I,sla_st,sla_nd,J,slb_st,slb_nd = ids[it,:]
        for jt in prange(sla_st,sla_nd):
            aa = (sla_nd-sla_st)
            for kt in range(slb_st,slb_nd):
                g_repeated[jt,kt] = g[I,J]/(aa*(slb_nd-slb_st))

defvjp(average_kernel,grad_average_kernel,None,None)    