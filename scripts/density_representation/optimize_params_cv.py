import argparse
import time

import sys
sys.path.insert(0,'/home/musil/git/ml_tools/')

from ml_tools.base import np,sp
from ase.io import read
from copy import copy
import pandas as pd
from ml_tools.utils import load_data,tqdm_cs,get_score
from ml_tools.models import KRRFastCV
from ml_tools.kernels import KernelPower
from ml_tools.split import EnvironmentalKFold

def get_sp_mapping(frames,sp):
    ii = 0
    fid2gids = {it:[] for it in range(len(frames))}
    for iframe,cc in enumerate(frames):
        for ss in cc.get_atomic_numbers():
            if ss == sp:
                fid2gids[iframe].append(ii)
                ii += 1
    return fid2gids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Get CV score using full covariance mat""")

    parser.add_argument("--rawsoaps", type=str, help="Name of the metadata file refering to full feature matrix")
    parser.add_argument("--Nfps", type=int, help="Number of pseudo input to take from the fps ids")
    parser.add_argument("--Xinit", type=str, help="Comma-separated list of initial parameters to optimize over")
    parser.add_argument("--Nfold", type=int, help="Number of folds for the CV")

    parser.add_argument("--prop", type=str, help="Path to the corresponding properties")
    parser.add_argument("--out", type=str, help="Path to the corresponding output")
    
    args = parser.parse_args()

    prop_fn = args.prop
    y_train = np.load(prop_fn)

    Xinit = map(float, args.Xinit.split(','))
    
    Nfps = args.Nfps
    Nfold = args.Nfold

    rawsoaps_fn = args.rawsoaps
    params,rawsoaps = load_data(rawsoaps_fn) 
    fps_ids = params['fps_ids']
    soap_params = params['soap_params']
    kernel_params = params['kernel_params']
    env_mapping = params['env_mapping']
    out_fn = args.out

    #############################################

    X_active = rawsoaps[fps_ids[:Nfps]]
    
    cv = EnvironmentalKFold(n_splits=Nfold,random_state=10,shuffle=True,mapping=env_mapping)
