
from .internal_interface import RawSoapInternal

try:
    from .quippy_interface import RawSoapQUIP
    has_quippy = True
except:
    has_quippy = False


