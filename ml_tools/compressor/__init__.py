try:
  from .filter import SymmetryFilter
except ImportError:
  pass

from .fps import FPSFilter
from .cur import CURFilter
from .powerspectrum_cov import CompressorCovarianceUmat,AngularScaler