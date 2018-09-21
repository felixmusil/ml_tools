import ase
import os
from os import listdir
from os.path import isfile, join
import ase
from ase.visualize import view
from ase.io import read, write
import math
from string import Template
from ..base import np,sp
from copy import copy,deepcopy
from glob import glob
import cPickle as pck
import commands
import sh,re


p = re.compile(r"\s+") # replace white spaces and \n with commas
def extract_shift_tensor(contrib_string):
    nn = contrib_string.find('\n')
    bb = p.sub(',',contrib_string[nn+2:])[1:-1]
    return np.array(bb.split(','),dtype=float).reshape((3,3))

def extract_shift_scalar(contrib_string):
    bb = p.sub(' ',contrib_string)
    return float(bb.split(':')[-1])

def extract_shift_contribution(contribution_name,fn):
    if contribution_name in ['core']:
        aa = sh.egrep(contribution_name+' sigma:',fn).split('\n')[:-1]
        ee = [cc*np.eye(3) for cc in map(extract_shift_scalar,aa)]
    elif contribution_name in ['Macroscopic shape']:
        aa = str(sh.egrep('-A 3',contribution_name,fn))
        ee = extract_shift_tensor(aa)
    elif contribution_name in ['bare','para','dia','para_oo','para_lq','Total']:
        aa = sh.grep('-A 3',contribution_name+' sigma:',fn).split('--\n')
        ee = map(extract_shift_tensor,aa)
    return ee

def get_isotropic(mat):
    return np.mean(mat.diagonal())
def get_sum(*args):
    sss = np.sum(args,axis=0)
    return sss

def get_shift_local(fn,is_scalar=True):
    core = extract_shift_contribution('core',fn)
    bare = extract_shift_contribution('bare',fn)
    para = extract_shift_contribution('para',fn)
    para_oo = extract_shift_contribution('para_oo',fn)
    para_lq = extract_shift_contribution('para_lq',fn)
    dia = extract_shift_contribution('dia',fn)

    if is_scalar is True:
        iso = []
        for a,b,c,d,e,f in zip(core, bare, para, dia, para_oo, para_lq):
            iso.append(get_isotropic(get_sum(a,b,c,d,e,f)))
        out = np.round(iso,decimals=2)
        return out
    else:
        tensor = []
        for a,b,c,d,e,f in zip(core, bare, para, dia, para_oo, para_lq):
            tensor.append(get_sum(a,b,c,d,e,f))
        return np.array(tensor)
def get_shift_maroscopic(fn,is_scalar=True):
    macro = extract_shift_contribution('Macroscopic shape',fn)
    if is_scalar is True:
        iso = get_isotropic(macro)
        out = np.array([np.round(iso,decimals=2)])
        return out
    else:
        return np.array(macro)
def get_shift_total(fn,is_scalar=True):
    aa = extract_shift_contribution('Total',fn)
    if is_scalar is True:
        total = map(get_isotropic,aa)
        out = np.round(total,decimals=2)
        return out
    else:
        tensor = np.array(aa)
        return tensor

    