from sklearn.utils.metaestimators import _safe_split
from ..base import np,sp,BaseEstimator
from ..utils import tqdm_cs

class CrossValidationScorer(BaseEstimator):
    def __init__(self,cv=None,estimator=None,score_func=None,estimator_params=None):
        #super(CrossValidationScorer, self).__init__()
        if cv is None and estimator is None:
            pass
        self.cv = cv
        if estimator_params is None:
            self.estimator = estimator
        elif isinstance(estimator,type): # if the estimator hasn't been instanciated 
            self.estimator = estimator(**estimator_params)
        else: # if it has been instancited, then overwrite the parameters with the new ones
            self.estimator = estimator
            self.estimator.set_params(estimator_params)
        
        if isinstance(score_func,dict):
            self.score_func = score_func
        else:
            self.score_func = dict(score=score_func)
        
        aa = self.estimator.get_params()
        aa.pop('trainer')
        self.params = dict(cv=self.cv.get_params(),estimator=aa,
                          score_type=self.score_func.keys())
        
    def fit(self,X,y):
        self.scores = {k:[] for k in self.score_func}
        self.predictions = []
        for train,test in self.cv.split(X):
            X_train,y_train = _safe_split(self.estimator,X,y,train)
            X_test, y_true = _safe_split(self.estimator,X,y,test,train_indices=train)
            self.estimator.fit(X_train,y_train)
            y_pred = self.estimator.predict(X_test)
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
    def get_params(self,deep=True):
        return self.params
    def pack(self):
        state = dict(scores=self.scores,score=self.score,params=self.get_params(),predictions=self.predictions)
        return state
    def unpack(self,state):
        self.scores = state['scores']
        self.score = state['score']
        self.predictions = state['predictions']
        err_m = 'params are not consistent {} != {}'.format(self.params,state['params'])
        assert self.params == state['params'],err_m
        
    def loads(self,state):
        self.scores = state['scores']
        self.score = state['score']
        self.params = state['params']
        self.predictions = state['predictions']

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
            
    
