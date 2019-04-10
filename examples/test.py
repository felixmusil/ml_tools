import numpy as np
import ase
from ase.io import read,write
from ase.visualize import view
from glob import glob
from copy import copy
from tqdm import tqdm_notebook
import cPickle as pck

import sys,os
sys.path.insert(0,'../')

from ml_tools.descriptors import RawSoapInternal
from ml_tools.models import FullCovarianceTrainer,SoRTrainer
from ml_tools.utils import get_mae,get_rmse,get_sup,get_spearman,get_score,load_pck
from ml_tools.split import KFold,LCSplit,ShuffleSplit
from ml_tools.model_selection import CrossValidationScorer
from ml_tools.model_selection import GridSearch
from ml_tools.compressor import FPSFilter

# filename of the molecular crystal structures
fn = './data/CSD500.xyz'
# atomic type for which to predict chemical shieldings
sp = 1
# atomic types present in the structures
global_species=[1, 6, 7, 8]
nocenters = copy(global_species)
nocenters.remove(sp)
# parameters for the soap descriptor
soap_params = dict(rc=4, nmax=8, lmax=6, awidth=0.4,
                   global_species=global_species,
                   nocenters=nocenters,
                   disable_pbar=True)

self_contribution = {1:0., 6:0., 7:0., 8:0.}

def get_sp_mapping(frames,sp):
    ii = 0
    fid2gids = {it:[] for it in range(len(frames))}
    for iframe,cc in enumerate(frames):
        for ss in cc.get_atomic_numbers():
            if ss == sp:
                fid2gids[iframe].append(ii)
                ii += 1
    return fid2gids

def extract_chemical_shielding(frames,sp):
    prop = []
    for cc in frames:
        numb = cc.get_atomic_numbers()
        prop.extend(cc.get_array('CS')[numb==sp])
    y = np.array(prop)
    return y

frames_train = read(fn,index=':10')
y_train = extract_chemical_shielding(frames_train,sp)

zeta = 2

representation = RawSoapInternal(**soap_params)

trainer = FullCovarianceTrainer(zeta=2, model_name='krr', kernel_name='power', self_contribution=self_contribution,has_global_targets=False,feature_transformations=representation)

# print trainer
# aa = trainer.to_dict()
# print aa
# bb = trainer.from_dict(aa)
# print bb.__dict__
# print bb.to_dict()


X_train = representation.transform(frames_train)
trainer.precompute(y_train, X_train)

model = trainer.fit(sigmas=[1e-2])


frames_test = read(fn,index='10:20')
y_test = extract_chemical_shielding(frames_test,sp)
print model.alpha
y_pred = model.predict(frames_test)
print get_score(y_pred,y_test)