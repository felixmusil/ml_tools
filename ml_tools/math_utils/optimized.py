from numba import njit,prange,void,float64,int32,vectorize
import math
# from ..base import np
# from autograd.extend import primitive,defvjp,defjvp

# @primitive
@vectorize([float64(float64, float64),float64(float64, int32)])
def power(x,zeta):
    return math.pow(x,zeta)

# def power_vjp(ans, x,zeta):
#     x_shape = x.shape
#     return lambda g: np.full(x_shape, g) * zeta * power(x,zeta-1) 
    
# defvjp(power, power_vjp)

# from autograd.numpy.numpy_vjps import unbroadcast_f,replace_zero

# defvjp(power,
#     lambda ans, x, y : unbroadcast_f(x, lambda g: g * y * x ** np.where(y, y - 1, 1.)),
#     lambda ans, x, y : unbroadcast_f(y, lambda g: g * np.log(replace_zero(x, 1.)) * x ** y))  

# defjvp(power,      
#     lambda g, ans, x, y : g * y * x ** np.where(y, y - 1, 1.),
#     lambda g, ans, x, y : g * np.log(replace_zero(x, 1.)) * x ** y)