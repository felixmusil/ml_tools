import argparse
import time

import sys,os
sys.path.insert(0,'/home/musil/git/ml_tools/')

from autograd import grad

from ml_tools.base import np,sp
 
from ml_tools.utils import load_data,tqdm_cs,get_score,dump_json,load_json,dump_pck,load_pck
from ml_tools.models import KRR,TrainerCholesky
from ml_tools.kernels import KernelPower,KernelSparseSoR
from ml_tools.split import EnvironmentalShuffleSplit,LCSplit,ShuffleSplit
from ml_tools.compressor import CompressorCovarianceUmat
from ml_tools.descriptors import RawSoapQUIP
import pandas as pd
from ase.io import read


EXPECTED_INPUT = dict(
  soap_params=dict(),
  base_kernel=dict(params=dict(zeta=2)),
  env_mapping_fn='',
  model=dict(params=dict(jitter=1e-8,delta=0.1)),
  sparse_kernel=dict(active_inp=dict(ids_fn='',Nfps=10),params=dict(Lambda=10)),
  input_data=dict(frames_fn='',feature_mat_fn='',Kmat_fn=''),
  lc_params=dict(n_repeats=[1],train_sizes=[1],test_size=10,random_state=10),
  prop_fn='',
  start_from_iter=0,
  out_fn=dict(scores='',results='')
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Get LC""")

    parser.add_argument("--input", type=str, help="Name of the metadata file containing the input data")
    
    args = parser.parse_args()

    inp = load_json(args.input)

    scores_fn = os.path.abspath(inp['out_fn']['scores'])
    results_fn = os.path.abspath(inp['out_fn']['results'])
    # inp['']
    prop_fn = os.path.abspath(inp['prop_fn'])
    y = np.load(prop_fn)
    if 'start_from_iter' in inp:
        start_from_iter = inp['start_from_iter']
    else:
        start_from_iter = 0
    if 'env_mapping_fn' in inp:
        env_mapping_fn = os.path.abspath(inp['env_mapping_fn'])
        env_mapping = load_json(env_mapping_fn)

    lc_params = inp['lc_params']
    input_data = inp['input_data']
    jitter = inp['model']['params']['jitter']

    if 'sparse_kernel' in inp:
        active_inp = inp['sparse_kernel']['active_inp']
        Lambda = inp['sparse_kernel']['params']['Lambda']
        active_ids = np.load(active_inp['ids_fn'])[:active_inp['Nfps']]
        Mactive = len(active_ids)
        delta = 1
        is_SoR = True
    else:
        delta = inp['model']['params']['delta']
        is_SoR = False

    if 'frames_fn' in input_data:
      frames_fn = os.path.abspath(input_data['frames_fn'])
      representation = RawSoapQUIP(**inp['soap_params'])
      compute_rep = True
      compute_kernel = True
    elif 'feature_mat_fn' in input_data:
      rawsoaps_fn = os.path.abspath(input_data['feature_mat_fn'])
      kernel = KernelPower(**inp['kernel_params'])
      compute_rep = False
      compute_kernel = True
    elif 'Kmat_fn' in input_data:
      Kmat_fn = os.path.abspath(input_data['Kmat_fn'])
      compute_rep = False
      compute_kernel = False
        
    #############################################

    print('Get representation')
    if compute_rep is True:
        frames = read(frames_fn,index=':')
        rawsoaps = representation.transform(frames)
        Nsample = len(rawsoaps)
    elif compute_rep is False and compute_kernel is True:
        params,rawsoaps = load_data(rawsoaps_fn,mmap_mode=None)
        Nsample = len(rawsoaps)

    print('Get kernel')
    if compute_kernel is True:
        if is_SoR is True:
            X_active = rawsoaps[active_ids]
            kMM = kernel(X_active,X_active)
            kMN = kernel(X_active,rawsoaps)
        else:
            Kmat = kernel.transform(rawsoaps)
        Nsample = kMN.shape[1]
    elif compute_rep is False and compute_kernel is False:
        if is_SoR is True:
            params,Kmat = load_data(Kmat_fn,mmap_mode=None)
            Nsample = Kmat.shape[1]
            kMM = Kmat[np.ix_(active_ids,active_ids)]
            kMN = Kmat[active_ids]
        else:
            params,Kmat = load_data(Kmat_fn,mmap_mode=None)
            Nsample = Kmat.shape[0]
  
   
    trainer = TrainerCholesky(memory_efficient=True)
    model = KRR(jitter,delta,trainer)
    lc = LCSplit(ShuffleSplit, **lc_params)

    scores = []
    results = dict(input_params=inp,results=[])
    ii = 0
    for train,test in tqdm_cs(lc.split(y.reshape((-1,1))),total=lc.n_splits,desc='LC'):
        if ii >= start_from_iter:
            if is_SoR is True:
                kMN_train =  kMN[:,train]
                k_train = kMM + np.dot(kMN_train,kMN_train.T)/Lambda**2
                y_train = np.dot(kMN_train,y[train])/Lambda**2
                k_test = kMN[:,test]
            else:
                k_train = Kmat[np.ix_(train,train)]
                y_train = y[train]
                k_test = Kmat[np.ix_(test,train)]

            alpha = np.linalg.solve(k_train, y_train).flatten()
            y_pred = np.dot(alpha,k_test).flatten()
            #model.fit(k_train,y_train)
            #y_pred = model.predict(k_test)

            sc = get_score(y_pred,y[test])
            dd = dict(Ntrain = len(train), Ntest = len(test),iter=ii)
            sc.update(**dd)
            scores.append(sc)
            
            results['results'].append(dict(y_pred=y_pred,y_true=y[test],iter=ii,
                                            Ntrain = len(train), Ntest = len(test)))
            
            df = pd.DataFrame(scores)
            df.to_json(scores_fn)
            dump_pck(results_fn,results)
        ii += 1

    