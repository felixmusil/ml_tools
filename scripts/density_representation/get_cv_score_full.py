import argparse
import time

import sys
sys.path.insert(0,'../../')

import ml_tools as ml 
from ml_tools.base import np,sp
from ase.io import read
from copy import copy
import pandas as pd

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

    parser.add_argument("--kernel", type=str, help="Name of the metadata file refering to full kernel matrix")
    parser.add_argument("--prop", type=str, help="Path to the corresponding properties")
    parser.add_argument("--out", type=str, help="Path to the corresponding properties")
    parser.add_argument("--deltas", type=str, default="",help="Comma-separated list of  (e.g. --nocenter 1,2,4)")
    
    args = parser.parse_args()

    prop_fn = args.prop
    y_train = np.load(prop_fn)

    deltas = map(float, args.deltas.split(','))
    deltas = sorted(list(set(deltas)))

    kernel_fn = args.kernel
    params,_ = ml.io_utils.load_data(kernel_fn) 
    
    out_fn = args.out

    #############################################

    fps_ids = params['fps_ids']
    soap_params = params['soap_params']
    kernel_params = params['kernel_params']
    env_mapping = params['env_mapping']

    kernel = ml.kernels.KernelPower(**kernel_params)

    cv = ml.split.EnvironmentalKFold(n_splits=10,random_state=10,shuffle=True,mapping=env_mapping)
    jitter = 1e-8

    scores = []
    preds = []
    
    for delta in ml.utils.tqdm_cs(deltas):
        krr = ml.modelsKRRFastCV(jitter,delta,cv)
        _,Kmat = ml.utils.load_data(kernel_fn,mmap_mode=None)
        krr.fit(Kmat,y_train)
        y_pred = krr.predict()
        sc = ml.utils.get_score(y_pred,y_train)
        sc.update(dict(delta=delta,y_pred=y_pred,y_true=y_true))
        scores.append(sc)
        preds.append(y_pred)

    df = pd.DataFrame(scores)

    df.to_json(out_fn)