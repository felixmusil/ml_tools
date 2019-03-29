from numba import njit,prange,void,float64,float32,int32,int64,vectorize,jit,config,threading_layer
import numpy as np
import math


@jit(nopython=True,nogil=True,cache=True)
def power_kernel(x,y,zeta):
    return math.pow(np.dot(x,y),zeta)

@njit(parallel=True)
def sum_power_no_species(out, zeta, desc1, ids1, desc2, ids2):
    N = ids1.shape[0]
    M = ids2.shape[0]
    for idesc1 in prange(N):
        iframe1, sp1 = ids1[idesc1]
        for idesc2 in prange(M):
            iframe2, sp2 = ids2[idesc2]
            out[iframe1,iframe2] += power_kernel(desc1[idesc1],desc2[idesc2],zeta)

@njit(parallel=True)
def sum_power_no_species_self(out, zeta, desc1, ids1):
    N = ids1.shape[0]
    for idesc1 in prange(N):
        iframe1, sp1 = ids1[idesc1]
        out[iframe1,iframe1] += power_kernel(desc1[idesc1],desc1[idesc1],zeta)
        for idesc2 in prange(idesc1+1,N):
            iframe2, sp2 = ids1[idesc2]
            tmp = power_kernel(desc1[idesc1],desc1[idesc2],zeta)
            out[iframe1,iframe2] += tmp
            out[iframe2,iframe1] += tmp

@njit(parallel=True)
def derivative_sum_power_no_species(out, zeta, desc1, grad1, ids1, desc2, ids2):
    N = ids1.shape[0]
    M = ids2.shape[0]
    for idesc2 in prange(M):
        iframe2, sp2 = ids2[idesc2]
        for igrad1 in prange(N):
            idesc1, ipos1, sp1 = ids1[igrad1]
            fac = np.dot(grad1[igrad1],desc2[idesc2])
            if zeta != 1:
                fac *= zeta * power_kernel(desc1[idesc1],desc2[idesc2],zeta-1)
            out[ipos1,:,iframe2] += fac

@njit(parallel=True)
def sum_power_diag(out, zeta, desc1, ids1):
    N = ids1.shape[0]
    for idesc1 in prange(N):
        iframe1, sp1 = ids1[idesc1]
        out[iframe1] += power_kernel(desc1[idesc1],desc1[idesc1],zeta)

@njit(parallel=True)
def sum_power_diff_species(out, zeta, desc1, ids1, desc2, ids2):
    N = ids1.shape[0]
    M = ids2.shape[0]
    for idesc1 in prange(N):
        iframe1, sp1 = ids1[idesc1]
        for idesc2 in prange(M):
            iframe2, sp2 = ids2[idesc2]
            if sp1 != sp2: continue
            out[iframe1,iframe2] += power_kernel(desc1[idesc1],desc2[idesc2],zeta)

@njit(parallel=True)
def sum_power_diff_species_self(out, zeta, desc1, ids1):
    N = ids1.shape[0]
    for idesc1 in prange(N):
        iframe1, sp1 = ids1[idesc1]
        out[iframe1,iframe1] += power_kernel(desc1[idesc1],desc1[idesc1],zeta)
        for idesc2 in prange(idesc1+1,N):
            iframe2, sp2 = ids1[idesc2]
            if sp1 != sp2: continue
            tmp = power_kernel(desc1[idesc1],desc1[idesc2],zeta)
            out[iframe1,iframe2] += tmp
            out[iframe2,iframe1] += tmp

@njit(parallel=True)
def derivative_sum_power_diff_species(out, zeta, desc1, grad1, ids1, desc2, ids2):
    N = ids1.shape[0]
    M = ids2.shape[0]
    for igrad1 in prange(N):
        idesc1, ipos1, sp1 = ids1[igrad1]
        for idesc2 in prange(M):
            iframe2, sp2 = ids2[idesc2]
            if sp1 != sp2: continue
            fac = np.dot(grad1[igrad1],desc2[idesc2])
            if zeta != 1:
                fac *= zeta * power_kernel(desc1[idesc1],desc2[idesc2],zeta-1)
            out[ipos1,:,iframe2] += fac