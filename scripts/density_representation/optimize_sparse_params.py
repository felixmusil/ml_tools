import ase
from ase.io import read,write
from ase.visualize import view
import sys,os
import mkl
from glob import glob
from copy import copy
from sklearn.model_selection import KFold,ParameterGrid
import cPickle as pck
import json
import pandas as pd

sys.path.insert(0,'/home/musil/git/ml_tools/')

from ml_tools.descriptors.quippy_interface import RawSoapQUIP
from ml_tools.models.KRR import KRR,TrainerCholesky,KRRFastCV
from ml_tools.models.pipelines import RegressorPipeline
from ml_tools.models.handlers import HashJsonHandler
from ml_tools.kernels.kernels import KernelPower,KernelSparseSoR,KernelSum
from ml_tools.utils import (get_mae,get_rmse,get_sup,get_spearman,
                            get_score,tqdm_cs,load_pck,dump_json,load_json,
                            load_data,dump_data)
from ml_tools.split import KFold,EnvironmentalKFold,LCSplit,ShuffleSplit,EnvironmentalShuffleSplit
from ml_tools.model_selection.scorer import CrossValidationScorer
from ml_tools.model_selection.gs import GridSearch
from ml_tools.base import KernelBase,BaseEstimator,TransformerMixin
from ml_tools.math_utils.optimized import power
from ml_tools.compressor.fps import FPSFilter
from ml_tools.compressor.filter import SymmetryFilter
from ml_tools.compressor.powerspectrum_cov import CompressorCovarianceUmat
from ml_tools.base import np,sp

mkl.set_num_threads(15)

path = '/home/musil/workspace/density_paper/'
xyzPath = path + 'data/'
fn_in = path + 'results/nmr/active_set/full_kernel_umat_sp_compression_0.json'
fn_prop = path + 'data/CSD890_H.npy'
fn_out = path + 'results/nmr/active_set/opt_cv_score_sparse.json'



y_train = np.load(fn_prop)
print fn_in
params, Kmat = load_data(fn_in,mmap_mode=None) 
fps_ids = params['fps_ids']
soap_params = params['soap_params']
kernel_params = params['kernel_params']
env_mapping = params['env_mapping']


kernel = KernelPower(**kernel_params)
#trainer = TrainerCholesky(memory_efficient=False)
cv = EnvironmentalKFold(n_splits=10,random_state=10,shuffle=True,mapping=env_mapping)
jitter = 1e-8

trainer = TrainerCholesky(memory_efficient=False)

scores = []

deltas = [1,1e-1,1e-2,1e-3]
Lambdas = [2,1,0.7,0.5,0.1]
N_active_samples = [3000,5000,10000,15000,20000]

for delta in tqdm_cs(deltas,desc='delta'):
    krr = KRR(jitter,delta,trainer)
    for N_active_sample in tqdm_cs(N_active_samples,desc='N_active_sample',leave=False):
        active_ids = fps_ids[:N_active_sample]
        kMM = Kmat[np.ix_(active_ids,active_ids)]
        for Lambda in tqdm_cs(Lambdas,desc='Lambda',leave=False):
            preds = []
            y_pred = np.zeros(y_train.shape)
            for train,test in tqdm_cs(cv.split(Kmat),desc='cv',total=cv.n_splits,leave=False):
                kMN = Kmat[np.ix_(active_ids,train)]
                ## assumes Lambda= Lambda**2*np.diag(np.ones(n))
                sparseK = kMM + np.dot(kMN,kMN.T)/Lambda**2
                sparseY = np.dot(kMN,y_train[train])
                Ktest = Kmat[np.ix_(test,active_ids)]
                krr.fit(sparseK,sparseY)
                y_pred[test] = krr.predict(Ktest)

            sc = get_score(y_pred,y_train)
            sc.update(dict(N_active_sample=N_active_sample,
                           delta=delta,Lambda=Lambda))
            
            print sc
            scores.append(sc)
            df = pd.DataFrame(scores)
            df.to_json(fn_out)
