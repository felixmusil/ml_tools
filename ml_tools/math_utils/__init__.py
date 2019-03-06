
try:
  from .optimized import (power,average_kernel,symmetrize,
                          get_unlin_soap,sum_kernel)
  is_optimized = True
except:
  from .basic import (power,average_kernel,symmetrize,
                        get_unlin_soap,sum_kernel)
  is_optimized = False

