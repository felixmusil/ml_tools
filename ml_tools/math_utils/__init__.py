is_optimized = False

try:
  from .optimized import power,average_kernel
  is_optimized = True
except:
  from .basic import power,average_kernel
   