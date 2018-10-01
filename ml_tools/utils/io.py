import os
import pickle as pck
from ..base import np,sp
from scipy.sparse import save_npz,load_npz

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
        json.dump(data,f,sort_keys=True,indent=2)

def load_json(fn):
    with open(fn,'r') as f:
        data = json.load(f)
    return data

def dump_data(fn,metadata,data,is_sparse=False,compressed=False):
    
    data_fn = os.path.join(os.path.dirname(fn),metadata['fn'])
    if is_sparse is False:
        np.save(data_fn,data)
    else:
        save_npz(data_fn,data,compressed=compressed)
    dump_json(fn,metadata)
    
def load_data(fn,mmap_mode='r',is_sparse=False):
    metadata = load_json(fn)
    data_fn = os.path.join(os.path.dirname(fn),metadata['fn'])
    if is_sparse is False:
        data = np.load(data_fn,mmap_mode=mmap_mode)
    else:
        data = load_npz(data_fn)
    return metadata,data



def check_file(fn):
    return os.path.isfile(fn)