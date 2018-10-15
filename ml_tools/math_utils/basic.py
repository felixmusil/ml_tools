from ..base import np,sp
from autograd.extend import primitive,defvjp,defjvp

def power(x,zeta):
    return np.power(x,zeta)

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
                g_repeated[ist:ind,jst:jnd] = g[I,J]/((ind-ist)*(jnd-jst))
        return g_repeated
    return vjp

defvjp(average_kernel,grad_average_kernel,None,None)