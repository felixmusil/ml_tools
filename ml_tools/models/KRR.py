from ..base import TrainerBase,RegressorBase
from ..base import np,sp
try:
    from autograd.scipy.linalg import cho_factor,cho_solve,solve_triangular
except:
    from scipy.linalg import cho_factor,cho_solve,solve_triangular

class KRR(RegressorBase):
    
    def __init__(self,jitter,reg):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter
        self.reg = reg
    
    def fit(self,kernel,y):
        '''Train the krr model with trainKernel and trainLabel.'''
        #sigma =  np.std(y)/np.sqrt(np.trace(kernel)/kernel.shape[0])/self.reg   
        sigma =  1./np.sqrt(np.trace(kernel)/kernel.shape[0])/self.reg

        L = np.linalg.cholesky(kernel+np.eye(kernel.shape[0])*sigma**2+np.eye(kernel.shape[0])*self.jitter)
        z = solve_triangular(L,y,lower=True)
        self.alpha = solve_triangular(L.T,z,lower=False,overwrite_b=True).reshape((1,-1))

    def predict(self,testkernel):
        '''kernel.shape is expected as (nPred,nTrain)'''
        return np.dot(testkernel,self.alpha.flatten()).reshape((-1))


class KRR_PP(RegressorBase):
    """
    Class which uses Kernel_PP to perform a projected process KRR 
    Andrea Anelli 2019 18/03
    """
    
    def __init__(self,jitter,reg):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter
        self.reg = reg
    
    def fit(self,kernelMM,kernelMN,y):
        '''Train the krr model with trainKernel and trainLabel.'''
        #sigma =  np.std(y)/np.sqrt(np.trace(kernelMM)/kernelMM.shape[0])/self.reg
        sigma =  1./np.sqrt(np.trace(kernelMM)/kernelMM.shape[0])/self.reg
        ym = np.dot(kernelMN,y)
        kernel = np.eye(kernelMM.shape[0])*sigma**2*kernelMM + np.dot(kernelMN,kernelMN.T)
        L = np.linalg.cholesky(kernel+np.eye(kernelMM.shape[0])*self.jitter)
        z = solve_triangular(L,ym,lower=True)
        self.alpha = solve_triangular(L.T,z,lower=False,overwrite_b=True).reshape((1,-1))

    def predict(self,testkernel):
        '''kernel.shape is expected as (nPred,nTrain)'''
        y_pp = np.dot(testkernel,self.alpha.flatten()).reshape((-1))
        return y_pp




class KRRFastCV(RegressorBase):
    """ 
    taken from:
    An, S., Liu, W., & Venkatesh, S. (2007). 
    Fast cross-validation algorithms for least squares support vector machine and kernel ridge regression. 
    Pattern Recognition, 40(8), 2154-2162. https://doi.org/10.1016/j.patcog.2006.12.015
    """
    _pairwise = True
    
    def __init__(self,jitter,delta,cv):
        self.jitter = jitter
        self.cv = cv
        self.delta = delta
    
    def fit(self,kernel,y):
        '''Fast cv scheme. Destroy kernel.'''
        invkernel = kernel.copy()
        np.multiply(self.delta**2,invkernel,out=invkernel)
        invkernel[np.diag_indices_from(invkernel)] += self.jitter
        invkernel = np.linalg.inv(invkernel)
        alpha = np.dot(invkernel,y)
        Cii = []
        beta = np.zeros(alpha.shape)
        self.y_pred = np.zeros(y.shape)
        self.error = np.zeros(y.shape)
        for _,test in self.cv.split(invkernel):
            Cii = invkernel[np.ix_(test,test)]
            beta = np.linalg.solve(Cii,alpha[test]) 
            self.y_pred[test] = y[test] - beta
            self.error[test] = beta # beta = y_true - y_pred 

        del invkernel
        
    def predict(self,kernel=None):
        '''kernel.shape is expected as (nPred,nTrain)'''
        #kernel = self.kernel(X,self.X_train)
        return self.y_pred

    def get_params(self,deep=True):
        return dict(sigma=self.jitter ,cv=self.cv)
        
    def set_params(self,params,deep=True):
        self.jitter  = params['jitter']
        self.cv = params['cv']
        self.delta = params['delta']
        self.y_pred = None

    def pack(self):
        state = dict(y_pred=self.y_pred,cv=self.cv.pack(),
                     jitter=self.jitter,delta=self.delta )
        return state
    def unpack(self,state):
        self.y_pred = state['y_pred']
        self.cv.unpack(state['cv'])
        self.delta = state['delta']

        err_m = 'jitter are not consistent {} != {}'.format(self.jitter ,state['jitter'])
        assert self.jitter  == state['jitter'], err_m
    def loads(self,state):
        self.y_pred = state['y_pred']
        self.cv.loads(state['cv'])
        self.jitter  = state['jitter']
        self.delta = state['delta']
