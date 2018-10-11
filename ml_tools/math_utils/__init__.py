is_optimized = False

try:
  from .optimized import power
  is_optimized = True
except:
  from .basic import power
   