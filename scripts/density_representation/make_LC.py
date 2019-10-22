# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time

import sys,os
sys.path.insert(0,'/home/musil/git/ml_tools/')


# from autograd import grad
from ml_tools.base import np,sp
# import numpy as np
# import scipy as sp

from ml_tools.utils import load_data,tqdm_cs,get_score,dump_json,load_json,dump_pck,load_pck
from ml_tools.models import KRR,TrainerCholesky
from ml_tools.kernels import KernelPower,KernelSparseSoR,KernelSum
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
  sum_kernel=dict(strides_fn=''),
  input_data=dict(frames_fn='',feature_mat_fn='',Kmat_fn=''),
  lc_params=dict(n_repeats=[1],train_sizes=[1],test_size=10,random_state=10),
  prop_fn='',
  start_from_iter=0,
  dtype='',
  compressor=dict(fn='',scaling_weights=[]),
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

    if 'dtype' in inp:
        dtype = inp['dtype']
    else:
        dtype = 'float64'

    y = np.asarray((np.load(prop_fn)),dtype=dtype)

    if 'start_from_iter' in inp:
        start_from_iter = inp['start_from_iter']
    else:
        start_from_iter = 0

    lc_params = inp['lc_params']
    input_data = inp['input_data']
    jitter = inp['model']['params']['jitter']

    # if 'env_mapping_fn' in inp:
    #     env_mapping_fn = os.path.abspath(inp['env_mapping_fn'])
    #     env_mapping = load_json(env_mapping_fn)
    #     shuffler = EnvironmentalShuffleSplit
    #     lc_params.update(**dict(mapping=env_mapping))
    # else:
    #     shuffler = ShuffleSplit
    shuffler = ShuffleSplit

    if 'frames_fn' in input_data:
      frames_fn = os.path.abspath(input_data['frames_fn'])
      representation = RawSoapQUIP(**inp['soap_params'])
      compute_rep = True
      compute_kernel = True
    elif 'feature_mat_fn' in input_data:
      rawsoaps_fn = os.path.abspath(input_data['feature_mat_fn'])
      params = load_json(rawsoaps_fn)
      kernel = KernelPower(**inp['base_kernel']['params'])
      compute_rep = False
      compute_kernel = True
    elif 'Kmat_fn' in input_data:
      Kmat_fn = os.path.abspath(input_data['Kmat_fn'])
      compute_rep = False
      compute_kernel = False

    if 'compressor' in inp:
        has_compressor = True
        compressor = CompressorCovarianceUmat()
        if 'compressor_state' in params:
            state = params['compressor_state']
            print('Load compressor from: {}'.format(rawsoaps_fn))
        else:
            compressor_fn = inp['compressor']['fn']
            state = load_json(compressor_fn)
            print('Load compressor from: {}'.format(compressor_fn))

        compressor.unpack(state)

        compressor.dtype = dtype

        if 'scaling_weights' in inp['compressor']:
            compressor.set_scaling_weights(inp['compressor']['scaling_weights'])
        compressor.to_reshape = True
    else:
        has_compressor = False

    has_sum_kernel = False
    if 'sparse_kernel' in inp and 'sum_kernel' not in inp:
        active_inp = inp['sparse_kernel']['active_inp']
        Lambda = inp['sparse_kernel']['params']['Lambda']
        active_ids = np.load(active_inp['ids_fn'])[:active_inp['Nfps']]
        is_SoR = True

    elif 'sparse_kernel' in inp and 'sum_kernel' in inp:
        active_inp = inp['sparse_kernel']['active_inp']
        Lambda = inp['sparse_kernel']['params']['Lambda']
        Nfps = inp['sparse_kernel']['active_inp']['Nfps']

        if 'fps_ids' in params:
            fps_ids_frames = params['fps_ids']
        else:
            fps_ids_frames = np.load(active_inp['ids_fn'])
        if 'strides' in params:
            strides = params['strides']
        else:
            strides = np.load(inp['sum_kernel']['strides_fn'])

        ids = [list(range(st,nd)) for st,nd in  zip(strides[:-1],strides[1:])]
        active_ids,active_strides = [],[0]
        for idx in fps_ids_frames[:Nfps]:
            active_ids.extend(ids[idx])
            nn = active_strides[-1]+len(ids[idx])
            active_strides.append(nn)
        has_sum_kernel = True
        is_SoR = True

    else:
        delta = inp['model']['params']['delta']
        is_SoR = False


    #############################################

    print('Getting representation')
    sys.stdout.flush()
    if compute_rep is True:
        frames = read(frames_fn,index=':')
        X = representation.transform(frames)
        Nsample = len(X)
    elif compute_rep is False and compute_kernel is True:
        params,X = load_data(rawsoaps_fn)
        # X = np.asarray(X,dtype=dtype)
        Nsample = len(X)

    if is_SoR is True and compute_kernel is True:
        X_active = X[active_ids]

    if has_compressor is True and compute_kernel is True:
        X = compressor.transform(X)
        X_active = compressor.transform(X_active)

    if has_sum_kernel is True and compute_kernel is True:
        kernel = KernelSum(kernel)
        X_active = dict(strides=active_strides,feature_matrix=X_active)
        X = dict(strides=strides,feature_matrix=X)

    print('Getting kernel')
    sys.stdout.flush()
    if compute_kernel is True:
        if is_SoR is True:
            kMM = kernel.transform(X_active,X_train=X_active)
            kMN = kernel.transform(X_active,X_train=X)
            Nsample = kMN.shape[1]
        else:
            Kmat = kernel.transform(X)
            Nsample = Kmat.shape[0]

    elif compute_rep is False and compute_kernel is False:
        if is_SoR is True:
            params,Kmat = load_data(Kmat_fn,mmap_mode=None)
            Nsample = Kmat.shape[1]
            kMM = Kmat[np.ix_(active_ids,active_ids)]
            kMN = Kmat[active_ids]
        else:
            params,Kmat = load_data(Kmat_fn,mmap_mode=None)
        Nsample = Kmat.shape[0]


    # trainer = TrainerCholesky(memory_efficient=True)
    # model = KRR(jitter,delta,trainer)
    lc = LCSplit(shuffler, **lc_params)

    scores = []
    results = dict(input_params=inp,results=[])
    ii = 0
    for train,test in tqdm_cs(lc.split(y.reshape((-1,1))),total=lc.n_splits,desc='LC'):
        if ii >= start_from_iter:
            if is_SoR is True:
                Mactive = kMN.shape[0]
                kMN_train =  kMN[:,train]
                k_train = kMM + np.dot(kMN_train,kMN_train.T)/Lambda**2 + np.diag(np.ones(Mactive))*jitter
                y_train = np.dot(kMN_train,y[train])/Lambda**2
                k_test = kMN[:,test]
            else:
                Ntrain = len(train)
                k_train = Kmat[np.ix_(train,train)] + np.diag(np.ones(Ntrain))*jitter
                y_train = y[train]
                k_test = Kmat[np.ix_(train,test)]

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


    print('Finished')
