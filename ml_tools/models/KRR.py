from ..base import TrainerBase,RegressorBase,FeatureBase
from ..base import np,sp,is_npy
from ..utils import return_deepcopy,is_structures
from ..split import ShuffleSplit
try:
    from autograd.scipy.linalg import cho_factor,cho_solve,solve_triangular
except:
    from scipy.linalg import cho_factor,cho_solve,solve_triangular

def train_krr(k,y):
    # k[np.diag_indices_from(k)] += jitter
    return np.linalg.solve(k, y)

class KRR(RegressorBase):
    _pairwise = True

    def __init__(self,delta=1.,kernel=None,X_train=None,representation=None,self_contribution=None):
        # super(KRR,self).__init__()
        RegressorBase.__init__(self)
        # Weights of the krr model
        self.alpha = None
        self.delta = delta
        self.kernel = kernel
        self.X_train = X_train
        self.representation = representation
        self.self_contribution = self_contribution

    def fit(self,kernel,y):
        '''\alpha = (kernel ) ^{-1} y '''
        self.alpha = train_krr(kernel, y)

    def _preprocess_input(self,X,eval_gradient):
        Natoms = None
        Y_formation = None

        if is_structures(X) is True:
            X = self.representation.transform(X)

        if isinstance(X,FeatureBase) is True:
            Natoms = self._get_Natoms(X,eval_gradient)
            Y_formation = self._get_energy_baseline(X)
            kernel = self.kernel.transform(X,self.X_train,(eval_gradient,False))
        elif is_npy(X) is True:
            kernel = X

        return kernel,Natoms,Y_formation

    def _get_energy_baseline(self,X):
        Y_formation = np.zeros(X.get_nb_sample())
        for iframe,sp in X.get_ids():
            Y_formation[iframe] += self.self_contribution[sp]
        return Y_formation

    def _get_Natoms(self,X,eval_gradient):
        Natoms = np.zeros(X.get_nb_sample(),dtype=int)
        for iframe,sp in X.get_ids():
            Natoms[iframe] += 1
        if eval_gradient is True:
            aa = []
            for n_atom in Natoms:
                aa.extend([n_atom]*n_atom)
            Natoms = np.array(aa).reshape((-1,1))

        return Natoms

    def predict(self,X,eval_gradient=False):
        '''y_{\star} = kernel_{\star}(\delta^2 \alpha)'''
        kernel,Natoms,Y_formation = self._preprocess_input(X,eval_gradient)
        alpha = self.alpha * self.delta**2
        if eval_gradient is False:
            return Y_formation + np.dot(kernel,alpha).reshape((-1))
        else:
            return np.dot(kernel,alpha).reshape((-1,3))


    def get_weigths(self):
        return self.alpha
    @return_deepcopy
    def get_params(self,deep=True):
        aa =  dict(jitter=self.jitter,
                    delta=self.delta,
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


class CommiteeKRR(RegressorBase):
    _pairwise = True

    def __init__(self,jitter,X_train,kernel,representation,train_ids=None,self_contribution=None):
        super(CommiteeKRR,self).__init__(jitter,kernel,X_train,representation,self_contribution)

        self.train_ids = train_ids

        if train_ids is None:
            self.is_sor = True
        else:
            self.is_sor = False

        self.subsampling_correction = None

        self.alphas = []

    def fit(self,kernel,y):
        '''Train the krr model with trainKernel and trainLabel.'''
        alpha = train_krr(kernel, y)
        self.alphas.append(alpha)

    def predict(self,X,eval_gradient=False):
        kernel,Natoms,Y_formation = self._preprocess_input(X,eval_gradient)

        n_models = len(self.alphas)
        n_test = kernel.shape[0]
        ypred = np.zeros((n_test, n_models))

        if eval_gradient is False:

            # predict on all training structures
            if self.is_sor is True:
                alphas = np.array(self.alphas)
                ypred = np.dot(kernel,alphas.T).reshape((n_test,n_models))
            elif self.is_sor is False:
                for imodel in range(n_models):
                    ypred[imodel] = np.dot(ktest[:,self.train_ids[imodel]],self.alphas[imodel]).reshape((-1))

            # final prediction before correction
            ypred_mean = Y_formation + np.mean(ypred,axis=1)
            # if all models agree no correction is made, if they disagree a correction is made
            ypred_std = self.subsampling_correction * np.std(ypred,axis=1)

            return ypred_mean, ypred_std
        else:
            raise NotImplementedError()

    @return_deepcopy
    def get_params(self,deep=True):
        aa =  super(CommiteeKRR,self).get_params()
        aa['train_ids'] = self.train_ids
        return aa

    @return_deepcopy
    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(
            weights=self.alphas,
            subsampling_correction=self.subsampling_correction,
        )
        return state

    def loads(self,data):
        self.alphas = data['weights']
        self.subsampling_correction = data['subsampling_correction']


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
