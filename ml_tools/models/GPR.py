from ..base import np,sp
from scipy.linalg import cho_factor,cho_solve
import os
from ..utils import make_new_dir

# NOT working properly

def dummy(a):
    return a

class GPR(object):
    def __init__(self,sigma=None,zeta=None,func=None,invfunc=None,memory_eff=False,dirname=None):
        if dirname is None:
            self.sigma = sigma
            self.zeta = zeta   
            if func is None or invfunc is None:
                self.func = dummy
                self.invfunc = dummy
            else:
                self.func = func
                self.invfunc = invfunc
            # Weights of the krr model
            self.alpha = None

            self.memory_eff = memory_eff
        else:
            self.load(dirname)
    def train(self,kernel,labels,ids=None,kernel_params=None):
        '''Train the krr model with trainKernel and trainLabel.'''
        
        nTrain, _ = kernel.shape
        self.kernel_params = kernel_params
        if ids is None:
            self.ids = np.arange(nTrain)
        else:
            self.ids = np.asarray(ids,dtype=int)
            # if kernel is a mem map the loading happens here
            kernel = kernel[np.ix_(self.ids,self.ids)]
            labels = np.asarray(labels[self.ids])
            
        # learn a function of the label
        trainLabel = self.func(labels)
        
        diag = kernel.diagonal().copy()
        self.lower = False
        reg = np.multiply(np.multiply(self.sigma ** 2, np.mean(diag)), np.var(trainLabel))
        # Effective regularization
        self.reg = reg
        if self.memory_eff:
            # kernel is modified here

            np.fill_diagonal(np.power(kernel, self.zeta, out=kernel),
                                np.add(np.power(diag,self.zeta,out=diag), reg,out=diag))

            kernel, self.lower = cho_factor(kernel, lower=False, overwrite_a=True, check_finite=False)
            self.L = kernel
        else:
            # kernel is not modified here
            reg = np.diag(reg)
            self.L, self.lower = cho_factor(np.power(kernel, self.zeta) + reg, lower=False, overwrite_a=False,check_finite=False)

        # set the weights of the krr model
        self.alpha = cho_solve((self.L, self.lower), trainLabel,overwrite_b=False).reshape((1,-1))
    def predict(self,kernel,self_kernel=None):
        '''kernel.shape is expected as (nTrain,nPred)'''
        
        
        if self.memory_eff:
            # kernel is modified in place here
            if np.abs(self.zeta-1) > 0:
                np.power(kernel,self.zeta,out=kernel)
            pred = self.invfunc(np.dot(self.alpha, kernel) ).reshape((-1))
            if self_kernel is not None:
                #kernel = cho_solve((self.L, self.lower), kernel,overwrite_b=True)
                aa = np.zeros((kernel.shape[1],))
                for ii in xrange(kernel.shape[1]):
                    aa[ii] = np.dot(kernel[:,ii],kernel[:,ii])
                #err2 = self.reg[0]*np.ones(aa.shape) + self_kernel - aa
                err2 = self_kernel - aa
           
        else:
            a_kernel = np.power(kernel,self.zeta)
            pred = self.invfunc(np.dot(self.alpha, a_kernel) ).reshape((-1))
            if self_kernel is not None:
                v = cho_solve((self.L, self.lower), a_kernel,overwrite_b=False)
                aa = np.zeros((v.shape[1],))
                for ii in xrange(v.shape[1]):
                    aa[ii] = np.dot(a_kernel[:,ii],v[:,ii])
                
                #err2 = self.reg[0]*np.ones(aa.shape) + self_kernel - aa
                err2 = self_kernel - aa
                
        if self_kernel is None:
            return pred
        else:    
            return pred,err2
            
    def save(self,fn):
        import os
        dirname = os.path.dirname(fn)
        new_dirname = make_new_dir(dirname)
        np.save(new_dirname+'/params.npy',dict(sigma=self.sigma,zeta=self.zeta,
                                             func=self.func,invfunc=self.invfunc,
                                             memory_eff=self.memory_eff,lower=self.lower,
                                             ids=self.ids,kernel_params=self.kernel_params))
        np.save(new_dirname+'/weights.npy',self.alpha)
        np.save(new_dirname+'/cho_factor.npy',self.L)
        
    def load(self,dirname):
        params = np.load(dirname+'params.npy').item()
        self.sigma,self.zeta,self.func,self.invfunc,\
            self.memory_eff,self.lower,self.ids,self.kernel_params = \
            params['sigma'],params['zeta'],params['func'],params['invfunc'], \
            params['memory_eff'],params['lower'],params['ids'],params['kernel_params']
        
        self.alpha = np.load(dirname+'weights.npy')
        self.L = np.load(dirname+'cho_factor.npy')
        
    