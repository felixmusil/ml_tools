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

      