from sklearn.base import BaseEstimator, RegressorMixin,TransformerMixin
import os
import importlib

try:
    import autograd.numpy as np
    import autograd.scipy as sp
except:
    import numpy as np
    import scipy as sp



CURRENT_VERSION = '1'

def is_npy(data):
    return isinstance(data, np.ndarray)

def is_large_array(data):
    if is_npy(data):
        if data.nbytes > 50e6:
            return True
        else:
            return False
    else:
        return False

def is_npy_filename(fn):
    if isinstance(fn,str):
        filename , file_extension = os.path.splitext(fn)
        if file_extension == '.npy':
            return True
        else:
            return False
    else:
        return False

def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def obj2dict_1(cls,state):
    VERSION = '1'
    module_name = cls.__module__
    class_name = cls.__name__
    frozen = dict(version=VERSION, class_name=class_name,
                    module_name=module_name,
                    init_params=state['init_params'],
                    data=state['data'])
    return frozen

def dict2obj_1(data):
    cls = get_class(data['module_name'],data['class_name'])
    obj = cls(**data['init_params'])
    obj.loads(data['data'])
    return obj

def is_valid_object_dict_1(data):
    valid_keys = ['version','class_name','module_name','init_params','data']
    aa = []
    if isinstance(data,dict):
        for k in data:
            if k in valid_keys:
                aa.append(True)
        if len(aa) == len(valid_keys):
            return True
        else:
            return False
    else:
        return False


obj2dict = {'1':obj2dict_1}
dict2obj = {'1':dict2obj_1}
is_valid_object_dict = {'1':is_valid_object_dict_1}

class BaseIO(object):
    def __init__(self):
        super(BaseIO,self).__init__()

    def to_dict(self,obj=None,version=CURRENT_VERSION):
        if obj is None:
            obj = self
        state = obj.dumps()

        for name,entry in state.items():
            if isinstance(entry, dict):
                for k,v in entry.items():
                    if isinstance(v, BaseIO) is True:
                         state[name][k] = self.to_dict(v,version)

        data = obj2dict[version](obj.__class__, state)
        return data

    def from_dict(self,data):
        version = data['version']

        for name,entry in data.items():
            if isinstance(entry, dict):
                for k,v in entry.items():
                    if is_valid_object_dict[version](v) is True:
                         data[name][k] = self.from_dict(v)

        obj = dict2obj[version](data)
        return obj

    def to_file(self,fn,version=CURRENT_VERSION):
        from .utils import dump_json
        fn = os.path.abspath(fn)
        filename , file_extension = os.path.splitext(fn)
        if file_extension == '.json':
            data = self.to_dict(version=version)
            self._dump_npy(fn,data)
            dump_json(fn,data)
        else:
            raise NotImplementedError('Unknown file extention: {}'.format(file_extension))

    def from_file(self,fn):
        from .utils import  load_json
        fn = os.path.abspath(fn)
        filename , file_extension = os.path.splitext(fn)
        if file_extension == '.json':
            data = load_json(fn)
            version = data['version']
            if is_valid_object_dict[version](data):
                self._load_npy(data)
                return self.from_dict(data)
            else:
                raise RuntimeError('The file: {}; does not contain a valid dictionary representation of an object.'.format(fn))

        else:
            raise NotImplementedError('Unknown file extention: {}'.format(file_extension))

    def _dump_npy(self,fn,data):
        filename , file_extension = os.path.splitext(fn)
        for k,v in data.items():
            if isinstance(v,dict):
                self._dump_npy(fn, v)
            elif is_large_array(v) is True:
                v_fn = filename + '-{}'.fomat(k) + '.npy'
                v_bfn = os.path.basename(v_fn)
                data[k] = v_bfn
                np.save(v_fn,v)

            elif is_npy(v)  is True:
                data[k] = v.tolist()

    def _load_npy(self,data):
        for k,v in data.items():
            if isinstance(v,dict):
                self._load_npy(v)
            elif is_npy_filename(v) is True:
                data[k] = np.load(v)





class RegressorBase(BaseIO, BaseEstimator, RegressorMixin):
    def __init__(self):
        super(RegressorBase,self).__init__()
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        pass
    def get_params(self,deep=True):
        pass
    def get_name(self):
        return type(self).__name__


class AtomicDescriptorBase(BaseIO, BaseEstimator,TransformerMixin):
    def __init__(self):
        super(AtomicDescriptorBase,self).__init__()
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        pass
    def get_params(self,deep=True):
        pass
    def get_name(self):
        return type(self).__name__

class TrainerBase(BaseIO):
    def __init__(self):
        super(TrainerBase,self).__init__()
    def fit(self):
        pass

class FeatureBase(BaseIO):
    def __init__(self):
        super(FeatureBase,self).__init__()

class CompressorBase(BaseIO,BaseEstimator,TransformerMixin):
    def __init__(self):
        super(CompressorBase,self).__init__()

class KernelBase(BaseIO, BaseEstimator,TransformerMixin):
    def __init__(self):
        super(KernelBase,self).__init__()
    def get_params(self):
        pass
    def set_params(self,**params):
        pass
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the kernel."""
    def get_name(self):
        return type(self).__name__


