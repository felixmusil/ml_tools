from builtins import range
from numba import njit,prange,void,float64,float32,int32,int64,vectorize,jit,config,threading_layer
import numpy as np
import math

@jit(nopython=True,nogil=True,parallel=True)
def get_strides(ids):
    subids = np.unique(ids[:,0])
    N = len(subids)
    Nelem = np.zeros((N+1,),np.int32)
    for ii in ids[:,0]:
        Nelem[ii+1] += 1
    Nelem = np.cumsum(Nelem)
    return N,Nelem

@jit(nopython=True,nogil=True,cache=True)
def power_kernel(x,y,zeta):
    return math.pow(np.vdot(x,y),zeta)

@njit(parallel=True)
def sum_power_no_species(out, zeta, desc1, ids1, desc2, ids2):
    N,Nelem = get_strides(ids1)
    M,Melem = get_strides(ids2)
    for iframe1 in prange(N):
        st,nd = Nelem[iframe1],Nelem[iframe1+1]
        d1 = desc1[st:nd]
        n = nd-st
        for iframe2 in prange(M):
            st2,nd2 = Melem[iframe2],Melem[iframe2+1]
            d2 = desc2[st2:nd2]
            m = nd2-st2
            for it in range(n):
                for jt in range(m):
                    out[iframe1,iframe2] += power_kernel(d1[it],d2[jt],zeta)
            out[iframe1,iframe2] /= n*m

@jit(nopython=True,nogil=True,parallel=True)
def sum_power_no_species_self(out, zeta, desc1, ids1):
    N,Nelem = get_strides(ids1)
    for iframe1 in prange(N):
        st,nd = Nelem[iframe1],Nelem[iframe1+1]
        d1 = desc1[st:nd]
        sp1 = ids1[st:nd,1]
        n = nd-st

        for it in range(n):
            for jt in range(n):
                out[iframe1,iframe1] += power_kernel(d1[it],d1[jt],zeta)
        out[iframe1,iframe1] /= n**2
        for iframe2 in range(iframe1+1,N):
            st2,nd2 = Nelem[iframe2],Nelem[iframe2+1]
            d2 = desc1[st2:nd2]
            sp2 = ids1[st2:nd2,1]
            m = nd2-st2
            for it in range(n):
                for jt in range(m):
                    tmp = power_kernel(d1[it],d2[jt],zeta)
                    out[iframe1,iframe2] += tmp
                    out[iframe2,iframe1] += tmp
            out[iframe1,iframe2] /= n*m
            out[iframe2,iframe1] /= n*m

@njit(parallel=True)
def derivative_sum_power_no_species(out, zeta, desc1, ids1, grad1, gids1, desc2, ids2):
    Ngrad = gids1.shape[0]

    Nc,Nelem = get_strides(gids1)
    Nf,fNelem = get_strides(ids1)
    M, Melem = get_strides(ids2)

    strides = np.zeros((Nc,4),np.int32)
    ii = 0
    for igrad in range(Ngrad):
        icenter, _, iframe , _ = gids1[igrad]
        if icenter != ii:
            ii = icenter
            strides[icenter] = np.array([Nelem[icenter],Nelem[icenter+1],
                                         fNelem[iframe],fNelem[iframe+1]],np.int32)
    for icenter in prange(Nc):
        st,nd,stf,ndf = strides[icenter]
        g1 = grad1[st:nd]
        gcids1 = gids1[st:nd]
        n_grad = nd - st
        ipos1, _, iframe1 , _ = gcids1[0]
        d1 = desc1[stf:ndf]
        for idesc2 in range(M):
            iframe2, sp2 = ids2[idesc2]
            d2 = desc2[idesc2]
            for igrad1 in range(n_grad):
                _, idesc1, _ , sp1 = gcids1[igrad1]
                fac = np.dot(g1[igrad1],d2)
                if zeta != 1:
                    fac *= zeta * power_kernel(d1[idesc1],d2,zeta-1)
                out[ipos1,:,iframe2] += fac

@njit(parallel=True)
def sum_power_diag(out, zeta, desc1, ids1):
    N = ids1.shape[0]
    for idesc1 in prange(N):
        iframe1, sp1 = ids1[idesc1]
        out[iframe1] += power_kernel(desc1[idesc1],desc1[idesc1],zeta)

@njit(parallel=True)
def power_diff_species(out, zeta, desc1, ids1, desc2, ids2):
    N = ids1.shape[0]
    M = ids2.shape[0]
    for idesc1 in prange(N):
        iframe1, sp1 = ids1[idesc1]
        for idesc2 in prange(M):
            iframe2, sp2 = ids2[idesc2]
            if sp1 != sp2: continue
            out[idesc1,idesc2] += power_kernel(desc1[idesc1],desc2[idesc2],zeta)


@jit(nopython=True,nogil=True,parallel=True)
def sum_power_diff_species(out, zeta, desc1, ids1, desc2, ids2):
    N,Nelem = get_strides(ids1)
    M,Melem = get_strides(ids2)
    for iframe1 in prange(N):
        st,nd = Nelem[iframe1],Nelem[iframe1+1]
        d1 = desc1[st:nd]
        sp1 = ids1[st:nd,1]
        n = nd-st
        for iframe2 in prange(M):
            st2,nd2 = Melem[iframe2],Melem[iframe2+1]
            d2 = desc2[st2:nd2]
            sp2 = ids2[st2:nd2,1]
            m = nd2-st2
            for it in range(n):
                for jt in range(m):
                    if sp1[it] != sp2[jt]: continue
                    out[iframe1,iframe2] += power_kernel(d1[it],d2[jt],zeta)

@jit(nopython=True,nogil=True,parallel=True)
def sum_power_diff_species_self(out, zeta, desc1, ids1):
    N,Nelem = get_strides(ids1)
    for iframe1 in prange(N):
        st,nd = Nelem[iframe1],Nelem[iframe1+1]
        d1 = desc1[st:nd]
        sp1 = ids1[st:nd,1]
        n = nd-st

        for it in range(n):
            for jt in range(n):
                if sp1[it] != sp1[jt]: continue
                out[iframe1,iframe1] += power_kernel(d1[it],d1[jt],zeta)

        for iframe2 in range(iframe1+1,N):
            st2,nd2 = Nelem[iframe2],Nelem[iframe2+1]
            d2 = desc1[st2:nd2]
            sp2 = ids1[st2:nd2,1]
            m = nd2-st2
            for it in range(n):
                for jt in range(m):
                    if sp1[it] != sp2[jt]: continue
                    tmp = power_kernel(d1[it],d2[jt],zeta)
                    out[iframe1,iframe2] += tmp
                    out[iframe2,iframe1] += tmp



@jit(nopython=True,nogil=True,parallel=True)
def derivative_sum_power_diff_species(out, zeta, desc1, ids1, grad1, gids1, desc2, ids2):
    Ngrad = gids1.shape[0]

    Nc,Nelem = get_strides(gids1)
    Nf,fNelem = get_strides(ids1)
    M, Melem = get_strides(ids2)

    strides = np.zeros((Nc,4),np.int32)
    ii = 0
    for igrad in range(Ngrad):
        icenter, _, iframe , _ = gids1[igrad]
        if icenter != ii:
            ii = icenter
            strides[icenter] = np.array([Nelem[icenter],Nelem[icenter+1],
                                         fNelem[iframe],fNelem[iframe+1]],np.int32)

    for icenter in prange(Nc):
        st,nd,stf,ndf = strides[icenter]
        g1 = grad1[st:nd]
        gcids1 = gids1[st:nd]
        n_grad = nd - st
        ipos1, _, iframe1 , _ = gcids1[0]
        d1 = desc1[stf:ndf]
        for idesc2 in range(M):
            iframe2, sp2 = ids2[idesc2]
            d2 = desc2[idesc2]
            for igrad1 in range(n_grad):
                _, idesc1, _ , sp1 = gcids1[igrad1]
                if sp1 != sp2: continue
                fac = np.dot(g1[igrad1],d2)
                if zeta != 1:
                    fac *= zeta * power_kernel(d1[idesc1],d2,zeta-1)
                out[ipos1,:,iframe2] += fac
