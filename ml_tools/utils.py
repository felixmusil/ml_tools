import os
from scipy.stats.mstats import spearmanr
from sklearn.metrics import r2_score
import numpy as np

def get_mae(ypred,y):
    return np.mean(np.abs(ypred-y))
def get_rmse(ypred,y):
    return np.sqrt(np.mean((ypred-y)**2))
def get_sup(ypred,y):
    return np.amax(np.abs((ypred-y)))
def get_spearman(ypred,y):
    corr,_ = spearmanr(ypred,y)
    return corr

def get_score(ypred,y):
    return get_mae(ypred,y),get_rmse(ypred,y),get_sup(ypred,y),r2_score(ypred,y),get_spearman(ypred,y)

  
def make_new_dir(fn):
    if not os.path.exists(fn):
        os.makedirs(fn)
    return fn