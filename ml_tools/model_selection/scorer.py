from ..base import np,sp,BaseEstimator
from ..utils import tqdm_cs,return_deepcopy,score_func
from ..models.trainers import FullCovarianceTrainer,SoRTrainer


class CrossValidationScorer(BaseEstimator):
    def __init__(self,trainer,cv,model_params):
        #super(CrossValidationScorer, self).__init__()
        self.model_params = model_params
        self.trainer = trainer
        self.cv = cv

        self.score_func = score_func

        self.y_pred = None
        self.mean_score = None
        self.error = None

    def fit(self,X,y):
        self.trainer.is_precomputed = False
        self.scores = {k:[] for k in self.score_func}
        self.predictions = []
        for train,test in self.cv.split(y[:,None]):
            model = self.trainer.fit(train_ids=train,y_train=y,X_train=X,**self.model_params)

            K_test,y_true = self.trainer.prepare_kernel_and_targets(train_ids=train,test_ids=test,**self.model_params)
            y_pred = model.predict(K_test)
            self.predictions.append(dict(y_ref=y_true,y_pred=y_pred))
            for k,func in self.score_func.iteritems():
                self.scores[k].append(func(y_true,y_pred))

        self.score = {k:np.mean(self.scores[k]) for k in self.score_func}

    def predict(self,X):
        raise 'CrossValidationScorer does not predict'

    def get_summary(self,txt=False):
        if txt is False:
            return dict(score=self.score,predictions=self.predictions)
        else:
            return ' '.join([k+'={:.3e}'.format(v) for k,v in self.score.iteritems()])

    @return_deepcopy
    def get_params(self,deep=True):
        return dict(model_params=self.model_params,trainer=self.trainer,cv=self.cv)

    @return_deepcopy
    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(y_pred=self.y_pred,mean_score=self.mean_score,error=self.error)
        return state

    def loads(self,state):
        self.y_pred = state['y_pred']
        self.mean_score = state['mean_score']
        self.error = state['error']

class KRRFastCVScorer(BaseEstimator):
    """
    Fast version for CV with full covariance KRR.

    taken from:
    An, S., Liu, W., & Venkatesh, S. (2007).
    Fast cross-validation algorithms for least squares support vector machine and kernel ridge regression.
    Pattern Recognition, 40(8), 2154-2162. https://doi.org/10.1016/j.patcog.2006.12.015
    """
    _pairwise = True

    def __init__(self,lambdas,trainer,cv):
        self.lambdas = lambdas
        self.trainer = trainer
        self.cv = cv

        self.score_func = score_func

        self.predictions = None
        self.scores = None
        self.score = None

    def fit(self,X,y):
        '''Fast cv scheme. Destroy kernel.'''
        self.trainer.is_precomputed = False
        kernel,y = self.trainer.prepare_kernel_and_targets(self.lambdas,y_train=y,X_train=X)

        kernel = np.linalg.inv(kernel)
        alpha = np.dot(kernel,y)
        Cii = []
        beta = np.zeros(alpha.shape)

        self.scores = {k:[] for k in self.score_func}
        self.predictions = []
        for _,test in self.cv.split(kernel):
            Cii = kernel[np.ix_(test,test)]
            beta = np.linalg.solve(Cii,alpha[test])
            y_pred = y[test] - beta
            y_true = y[test]
            self.predictions.append(dict(y_ref=y_true,y_pred=y_pred))
            for k,func in self.score_func.iteritems():
                self.scores[k].append(func(y_true,y_pred))

        self.score = {k:np.mean(self.scores[k]) for k in self.score_func}

        kernel = None

    def get_summary(self,txt=False):
        if txt is False:
            return dict(score=self.score,predictions=self.predictions)
        else:
            return ' '.join([k+'={:.3e}'.format(v) for k,v in self.score.iteritems()])

    def predict(self,kernel=None):
        raise 'CrossValidationScorer does not predict'

    def get_params(self,deep=True):
        return dict(lambdas=self.lambdas ,cv=self.cv,trainer=self.trainer)

    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(predictions=self.predictions,
                            scores=self.scores,
                            score=self.score)

    def loads(self,state):
        self.predictions = state['predictions']
        self.scores = state['scores']
        self.score = state['score']

class SoRCrossValidationScorer(CrossValidationScorer):
    def __init__(self,cv=None,score_func=None,estimator_params=None):
        # super(SoRCrossValidationScorer, self).__init__(
        #         cv=cv,score_func=score_func)
        self.cv = cv
        self.score_func = score_func
        self.Lambda = estimator_params['Lambda']
        self.jitter = estimator_params['jitter']

    def fit(self,X,y):
        kMM,kMN = X[0],X[1]
        self.scores = {k:[] for k in self.score_func}
        self.predictions = []
        Mactive,Nsample = kMN.shape

        for train,test in tqdm_cs(self.cv.split(kMN.T),desc='CV score',total=self.cv.n_splits):
            # prepare SoR kernel
            kMN_train =  kMN[:,train]
            kernel_train = kMM + np.dot(kMN_train,kMN_train.T)/self.Lambda**2 + np.diag(np.ones(Mactive))*self.jitter
            y_train = np.dot(kMN_train,y[train])/self.Lambda**2

            # train the KRR model
            alpha = np.linalg.solve(kernel_train, y_train).flatten()

            # make predictions
            kernel_test = kMN[:,test]
            y_true = y[test]
            y_pred = np.dot(alpha,kernel_test).flatten()

            self.predictions.append(dict(y_ref=y_true,y_pred=y_pred))
            for k,func in self.score_func.iteritems():
                self.scores[k].append(func(y_true,y_pred))

        self.score = {k:np.mean(self.scores[k]) for k in self.score_func}


