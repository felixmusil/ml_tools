import argparse
import time

import sys,os
sys.path.insert(0,'/home/musil/git/ml_tools/')

from autograd import grad

from ml_tools.base import np,sp

from ml_tools.utils import load_data,tqdm_cs,get_score,dump_json,load_json,load_pck
from ml_tools.models import KRRFastCV
from ml_tools.kernels import KernelPower
from ml_tools.split import EnvironmentalKFold
from ml_tools.compressor import CompressorCovarianceUmat

 
def get_sp_mapping(frames,sp):
    ii = 0
    fid2gids = {it:[] for it in range(len(frames))}
    for iframe,cc in enumerate(frames):
        for ss in cc.get_atomic_numbers():
            if ss == sp:
                fid2gids[iframe].append(ii)
                ii += 1
    return fid2gids

def optimize_loss(loss_func,x_start=None,args=None,maxiter=100,ftol=1e-6):
    from scipy.optimize import minimize
    gloss_func = grad(loss_func,argnum=0)
    
    pbar = tqdm_cs(total=maxiter)
    def call(Xi):
        Nfeval = pbar.n
        sss = ''
        for x in Xi:
            sss += '  '+'{:.5e}'.format(x)
        print('{0:4d}'.format(Nfeval) + sss)
        sys.stdout.flush()
        pbar.update()
    #const = spop.Bounds(np.zeros(x_start.shape),np.inf*np.ones(x_start.shape))
    myop = minimize(loss_func, x_start, args = args, jac=gloss_func,callback=call,
                      method = 'L-BFGS-B', options = {"maxiter": maxiter, "disp": False, "maxcor":9, 
                                                      "gtol":1e-9, "ftol":  ftol })
    pbar.close()
    return myop

def sor_loss(x_opt,X,y,cv,jitter,disable_pbar=True,leave=False,return_score=False):
    Lambda = x_opt[0]
    
    kMM = X[0]
    kMN = X[1]
    Mactive,Nsample = kMN.shape
    
    mse = 0
    y_p = np.zeros((Nsample,))
    for train,test in tqdm_cs(cv.split(kMN.T),total=cv.n_splits,disable=disable_pbar,leave=False):
        # prepare SoR kernel
        kMN_train =  kMN[:,train]
        kernel_train = kMM + np.dot(kMN_train,kMN_train.T)/Lambda**2 + np.diag(np.ones(Mactive))*jitter
        y_train = np.dot(kMN_train,y[train])/Lambda**2
        
        # train the KRR model
        alpha = np.linalg.solve(kernel_train, y_train).flatten()
        
        # make predictions
        kernel_test = kMN[:,test]
        y_pred = np.dot(alpha,kernel_test).flatten()
        if return_score is True:
            y_p[test] = y_pred
        
        mse += np.sum((y_pred-y[test])**2) 
    mse /= len(y)
    
    if return_score is True:
        score = get_score(y_p,y)
        return score
    return mse

def soap_cov_loss(x_opt,rawsoaps,y,cv,jitter,disable_pbar=True,leave=False,compressor=None,active_ids=None,return_score=False):
    Lambda = x_opt[0]
    fj = x_opt[1:]
    
    compressor.set_fj(fj)
    
    X = compressor.transform(rawsoaps)
    X_pseudo = X[active_ids]
    
    kMM = np.dot(X_pseudo,X_pseudo.T)
    kMN = np.dot(X_pseudo,X.T)
    Mactive,Nsample = kMN.shape
    
    mse = 0
    y_p = np.zeros((Nsample,))
    for train,test in tqdm_cs(cv.split(rawsoaps),total=cv.n_splits,disable=disable_pbar,leave=False):
        # prepare SoR kernel
        kMN_train =  kMN[:,train]
        kernel_train = (kMM + np.dot(kMN_train,kMN_train.T)/Lambda**2) + np.diag(np.ones(Mactive))*jitter
        y_train = np.dot(kMN_train,y[train])/Lambda**2
        
        # train the KRR model       
        alpha = np.linalg.solve(kernel_train, y_train).flatten()
        
        # make predictions
        kernel_test = kMN[:,test]
        y_pred = np.dot(alpha,kernel_test).flatten()
        if return_score is True:
            y_p[test] = y_pred
        
        mse += np.sum((y_pred-y[test])**2) 
    mse /= len(y)
    
    if return_score is True:
        score = get_score(y_p,y)
        return score
    
    return mse

def sor_fj_loss(x_opt,data,y,cv,jitter,disable_pbar=True,leave=False,kernel=None,compressor=None,return_score=False):
    Lambda = x_opt[0]
    fj = x_opt[1:]
    
    compressor.set_fj(fj)
    
    unlinsoaps = data[0]
    unlinsoaps_active = data[1]
    X = compressor.transform(unlinsoaps)
    X_active = compressor.transform(unlinsoaps_active)
    
    # kMM = kernel(X_active,X_active)
    # kMN = kernel(X_active,X)
    # TODO generalize to other kernels
    zeta = kernel.zeta
    kMM = np.power(np.dot(X_active,X_active.T),zeta)
    kMN = np.power(np.dot(X_active,X.T),zeta)
    Mactive,Nsample = kMN.shape
    
    mse = 0
    y_p = np.zeros((Nsample,))
    for train,test in tqdm_cs(cv.split(X),total=cv.n_splits,disable=disable_pbar,leave=False):
        # prepare SoR kernel
        kMN_train =  kMN[:,train]
        kernel_train = (kMM + np.dot(kMN_train,kMN_train.T)/Lambda**2) + np.diag(np.ones(Mactive))*jitter
        y_train = np.dot(kMN_train,y[train])/Lambda**2
        
        # train the KRR model       
        alpha = np.linalg.solve(kernel_train, y_train).flatten()
        
        # make predictions
        kernel_test = kMN[:,test]
        y_pred = np.dot(alpha,kernel_test).flatten()
        if return_score is True:
            y_p[test] = y_pred
        
        mse += np.sum((y_pred-y[test])**2) 
    mse /= len(y)
    
    if return_score is True:
        score = get_score(y_p,y)
        return score
    
    return mse

def LL_sor_loss(x_opt,X,y,cv,jitter,disable_pbar=True,leave=False,return_score=False):
    Lambda = x_opt[0]
    
    kMM = X[0]
    kMN = X[1]
    Mactive,Nsample = kMN.shape
    
    kernel_ = kMM + np.dot(kMN,kMN.T)/Lambda**2 + np.diag(np.ones(Mactive))*jitter
    y_ = np.dot(kMN,y)/Lambda**2
    
    # Get log likelihood score
    L = np.linalg.cholesky(kernel_)
    z = sp.linalg.solve_triangular(L,y_,lower=True)
    alpha = sp.linalg.solve_triangular(L.T,z,lower=False,overwrite_b=True).flatten()
    #alpha = np.linalg.solve(kernel_train, y_train).flatten()
    diag = np.zeros((Mactive))
    for ii in range(Mactive): diag[ii] = L[ii,ii]
    logL = -0.5* Mactive * np.log(2*np.pi) - 0.5 * np.vdot(y_,alpha) - np.sum(np.log(diag))
    return logL


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Get CV score using full covariance mat""")

    parser.add_argument("--X", type=str, help="Name of the metadata file refering to the input data")
    parser.add_argument("--Nfps", type=int,default=1, help="Number of pseudo input to take from the fps ids")
    parser.add_argument("--Xinit", type=str, help="Comma-separated list of initial parameters to optimize over")
    parser.add_argument("--Nfold", type=int, help="Number of folds for the CV")
    parser.add_argument("--jitter", type=float,default=1e-8, help="Jitter for numerical stability of Cholesky")
    parser.add_argument("--loss", type=str, help="Name of the bjective function to optimize with. Possible loss: sor_loss, soap_cov_loss, sor_fj_loss")
    parser.add_argument("--compressor", type=str,default='', help="Name of the json file containing the trained compressor data.")
    parser.add_argument("--ftol", type=float,default=1e-6, help="Relative tolerance for the optimization to exit: (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol ")
    parser.add_argument("--maxiter", type=int,default=100, help="Max Number of optimization steps")
    
    parser.add_argument("--prop", type=str, help="Path to the corresponding properties")
    parser.add_argument("--out", type=str, help="Path to the corresponding output")
    
    in_args = parser.parse_args()

    prop_fn = os.path.abspath(in_args.prop)
    y = np.load(prop_fn)

    
    x_init = map(float, in_args.Xinit.split(','))
    
    Nfps = in_args.Nfps
    Nfold = in_args.Nfold
    jitter = in_args.jitter
    maxiter = in_args.maxiter
    ftol = in_args.ftol

    rawsoaps_fn = os.path.abspath(in_args.X)
    print('Load data from: {}'.format(rawsoaps_fn))
    params,X = load_data(rawsoaps_fn,mmap_mode=None) 
    if 'fps_ids' in params:
        fps_ids = params['fps_ids']
    soap_params = params['soap_params']
    kernel_params = params['kernel_params']
    env_mapping = params['env_mapping']
    out_fn = in_args.out
    loss_type = in_args.loss 
    
    if len(in_args.compressor) > 0:
        compressor_fn = in_args.compressor
        print('Load compressor from: {}'.format(compressor_fn))
        compressor = CompressorCovarianceUmat()
        state = load_pck(compressor_fn)
        compressor.unpack(state)
        compressor.to_reshape = False
    else:
        compressor_fn = None
    #############################################
    
    cv = EnvironmentalKFold(n_splits=Nfold,random_state=10,shuffle=True,mapping=env_mapping)
    kernel = KernelPower(**kernel_params)
    print('Start: {}'.format(time.ctime()))
    if loss_type == 'sor_loss':
        loss_func = sor_loss
        X_active = X[fps_ids[:Nfps]]
        kMM = kernel(X_active,X_active)
        kMN = kernel(X_active,X)
        args = ((kMM,kMN),y,cv,jitter,False,False)
        if len(x_init) != 1:
            raise ValueError('with {} loss function, Xinit should have 1 argument'.format(loss_type))
    elif loss_type == 'soap_cov_loss':
        loss_func = soap_cov_loss
        active_ids = fps_ids[:Nfps]
        args = (X,y,cv,jitter,False,False,compressor,active_ids)
    elif loss_type == 'sor_fj_loss':
        loss_func = sor_fj_loss
        rawsoaps = X[0]
        rawsoaps_active = X[1]
        data = (compressor.reshape_(rawsoaps),compressor.reshape_(rawsoaps_active))
        args = (data,y,cv,jitter,False,False,kernel,compressor)
    else:
        raise ValueError('loss function: {}, does not exist.'.format(loss_type))

    print('Start optimization with {}'.format(x_init))
    sys.stdout.flush()
    x_opt = optimize_loss(loss_func,x_start=x_init,args=args,maxiter=maxiter,ftol=ftol)
    
    print('Optimized params:')
    print('{}'.format(x_opt))
    print('score with the optimized parameters:')
    new_args = [x_opt.x] + list(args) + [True]
    score = loss_func(*new_args)
    print('Score: {}'.format(score))

    data = dict(x_opt=x_opt.x.tolist(),score=score,x_init=x_init,
                maxiter=maxiter,ftol=ftol,loss_type=loss_type,Nfps=Nfps,
                compressor_fn=compressor_fn,
                Nfold=Nfold,rawsoaps_fn=rawsoaps_fn,prop_fn=prop_fn,message=x_opt.message)
    
    print('dump results in {}'.format(out_fn))
    dump_json(out_fn,data)