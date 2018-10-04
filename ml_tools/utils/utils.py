import os
from ..base import np,sp
from scipy.stats import spearmanr

def is_notebook():
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm import tqdm_notebook as tqdm_cs
    ascii = False
else:
    from tqdm import tqdm as tqdm_cs
    ascii = True

def get_r2(y_pred,y_true):
    weight = 1
    sample_weight = None
    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,dtype=np.float64)
    output_scores = 1 - (numerator / denominator)
    return np.mean(output_scores)

def get_mae(ypred,y):
    return np.mean(np.abs(ypred-y))
def get_rmse(ypred,y):
    return np.sqrt(np.mean((ypred-y)**2))
def get_sup(ypred,y):
    return np.amax(np.abs((ypred-y)))
def get_spearman(ypred,y):
    corr,_ = spearmanr(ypred,y)
    return corr

score_func = dict(
    MAE=get_mae,
    RMSE=get_rmse,
    SUP=get_sup,
    R2=get_r2,
    CORR=get_spearman
)

def get_score(ypred,y):
    scores = {}
    for k,func in score_func.iteritems():
        scores[k] = func(ypred,y)
    return scores

  
def make_new_dir(fn):
    if not os.path.exists(fn):
        os.makedirs(fn)
    return fn



def s2hms(time):
    m = time // 60
    s = int(time % 60)
    h = int(m // 60)
    m = int(m % 60)
    return '{:02d}:{:02d}:{:02d} (h:m:s)'.format(h,m,s)


def qp2ase(qpatoms):
    from ase import Atoms as aseAtoms
    positions = qpatoms.get_positions()
    cell = qpatoms.get_cell()
    numbers = qpatoms.get_atomic_numbers()
    pbc = qpatoms.get_pbc()
    atoms = aseAtoms(numbers=numbers, cell=cell, positions=positions, pbc=pbc)

    for key, item in qpatoms.arrays.iteritems():
        if key in ['positions', 'numbers', 'species', 'map_shift', 'n_neighb']:
            continue
        atoms.set_array(key, item)

    return atoms

def ase2qp(aseatoms):
    from quippy import Atoms as qpAtoms
    positions = aseatoms.get_positions()
    cell = aseatoms.get_cell()
    numbers = aseatoms.get_atomic_numbers()
    pbc = aseatoms.get_pbc()
    return qpAtoms(numbers=numbers,cell=cell,positions=positions,pbc=pbc)


    
