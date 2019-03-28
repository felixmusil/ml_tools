from ..base import TrainerBase,RegressorBase,FeatureBase
from ..base import np,sp
from ..utils import return_deepcopy
try:
    from autograd.scipy.linalg import cho_factor,cho_solve,solve_triangular
except:
    from scipy.linalg import cho_factor,cho_solve,solve_triangular

class KRR(RegressorBase):
    _pairwise = True

    def __init__(self,jitter,mean=None,kernel=None,X_train=None,representation=None):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter
        self.kernel = kernel
        self.X_train = X_train
        self.mean = mean
        self.representation = representation

    def fit(self,kernel,y):
        '''Train the krr model with trainKernel and trainLabel.'''
        reg = np.ones((kernel.shape[0],))*self.jitter
        alpha = np.linalg.solve(kernel+np.diag(reg), y)
        self.alpha = alpha

    def predict(self,X,eval_gradient=False):
        if isinstance(X,FeatureBase) is False:
            X = self.representation.transform(X)
        kernel = self.kernel.transform(X,self.X_train,(eval_gradient,False))
        if eval_gradient is False:
            return self.mean + np.dot(kernel,self.alpha).reshape((-1))
        else:

            return np.dot(kernel,self.alpha).reshape((-1,3))

    @return_deepcopy
    def get_params(self,deep=True):
        aa =  dict(jitter=self.jitter,
                    mean=self.mean,
                    kernel=self.kernel,
                    X_train=self.X_train,
                    representation=self.representation)
        return aa

    @return_deepcopy
    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(
            weights=self.alpha
        )
        return state

    def loads(self,data):
        self.alpha = data['weights']



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

    def fit(self,kernel,y):
        '''Fast cv scheme. Destroy kernel.'''
        np.multiply(self.delta**2,kernel,out=kernel)
        kernel[np.diag_indices_from(kernel)] += self.jitter
        kernel = np.linalg.inv(kernel)
        alpha = np.dot(kernel,y)
        Cii = []
        beta = np.zeros(alpha.shape)
        self.y_pred = np.zeros(y.shape)
        self.error = np.zeros(y.shape)
        for _,test in self.cv.split(kernel):
            Cii = kernel[np.ix_(test,test)]
            beta = np.linalg.solve(Cii,alpha[test])
            self.y_pred[test] = y[test] - beta
            self.error[test] = beta # beta = y_true - y_pred

        del kernel

    def predict(self,kernel=None):
        '''kernel.shape is expected as (nPred,nTrain)'''
        #kernel = self.kernel(X,self.X_train)
        return self.y_pred

    def get_params(self,deep=True):
        return dict(sigma=self.jitter ,cv=self.cv)

    def set_params(self,params,deep=True):
        self.jitter  = params['jitter']
        self.cv = params['cv']
        self.y_pred = None

    def pack(self):
        state = dict(y_pred=self.y_pred,cv=self.cv.pack(),
                     jitter=self.jitter)
        return state
    def unpack(self,state):
        self.y_pred = state['y_pred']
        self.cv.unpack(state['cv'])

        err_m = 'jitter are not consistent {} != {}'.format(self.jitter ,state['jitter'])
        assert self.jitter  == state['jitter'], err_m
    def loads(self,state):
        self.y_pred = state['y_pred']
        self.cv.loads(state['cv'])
        self.jitter  = state['jitter']
