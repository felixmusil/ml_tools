import warnings

try:
    from .internal_interface import RawSoapInternal
    has_internal_soap = True
except ImportError as e:
    warnings.warn("Could not import internal descriptor: {} {}".format(e.errno, e.strerror),category=ImportWarning)
    has_internal_soap = False
try:
    from .quippy_interface import RawSoapQUIP
    has_quippy = True
except ImportError as e:
    warnings.warn("Could not import quippy: {} {}".format(e.errno, e.strerror),category=ImportWarning)
    has_quippy = False

