import warnings

try:
    from .internal_interface import RawSoapInternal
    has_internal_soap = True
except ImportError:
    warnings.warn("Could not import internal descriptor",category=ImportWarning)
    has_internal_soap = False
try:
    from .quippy_interface import RawSoapQUIP
    has_quippy = True
except ImportError:
    warnings.warn("Could not import quippy",category=ImportWarning)
    has_quippy = False

