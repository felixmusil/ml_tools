import os
import pickle as pck
from ..base import np

try:
  import ujson as json
except:
  import json

def check_dir(workdir):
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    return workdir

def dump_pck(fn,data):
    with open(fn,'wb') as f:
        pck.dump(data,f,protocol=pck.HIGHEST_PROTOCOL)

def load_pck(fn):
    with open(fn,'rb') as f:
        data = pck.load(f)
    return data

def dump_json(fn,data):
    with open(fn,'w') as f:
        json.dump(data,f)

def load_json(fn):
    with open(fn,'r') as f:
        data = json.load(f)
    return data

def dump_data(fn,metadata,data):
    data_fn = metadata['fn']
    np.save(data_fn,data)
    dump_json(fn,metadata)
    
def load_data(fn,mmap_mode='r'):
    metadata = load_json(fn)
    data_fn = metadata['fn']
    data = np.load(data_fn,mmap_mode=mmap_mode)
    return metadata,data

def check_file(fn):
    return os.path.isfile(fn)