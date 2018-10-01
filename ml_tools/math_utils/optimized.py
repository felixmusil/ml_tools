from numba import njit,prange,void,float64,int32,vectorize
import math

@vectorize([float64(float64, float64),float64(float64, int32)])
def power(x,zeta):
    return math.pow(x,zeta)

