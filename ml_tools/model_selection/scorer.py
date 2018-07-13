from sklearn.utils.metaestimators import _safe_split
import numpy as np

class CrossValidationScorer(object):
    def __init__(self,cv=None,estimator=None,score_func=None):
        super(CrossValidationScorer, self).__init__()
        if cv is None and estimator is None:
            pass
        self.cv = cv
        self.estimator = estimator
        
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
        self.split_pred = []
        for train,test in self.cv.split(X):
            X_train,y_train = _safe_split(self.estimator,X,y,train)
            X_test, y_true = _safe_split(self.estimator,X,y,test,train_indices=train)
            self.estimator.fit(X_train,y_train)
            y_pred = self.estimator.predict(X_test)
            self.split_pred.append(dict(y_ref=y_true,y_pred=y_pred))
            for k,func in self.score_func.iteritems():
                self.scores[k].append(func(y_true,y_pred))
            
        self.score = {k:np.mean(self.scores[k]) for k in self.score_func}
            
    def predict(self,X):
        raise 'CrossValidationScorer does not predict'
    
    def get_summary(self,txt=False):
        if txt is False:
            return dict(score=self.score,predictions=self.split_pred)
        else:
            return ' '.join([k+'={:.3e}'.format(v) for k,v in self.score.iteritems()])
    def get_params(self,deep=True):
        return self.params
    def pack(self):
        state = dict(scores=self.scores,score=self.score,params=self.get_params(),predictions=self.split_pred)
        return state
    def unpack(self,state):
        self.scores = state['scores']
        self.score = state['score']
        self.split_pred = state['split_pred']
        err_m = 'params are not consistent {} != {}'.format(self.params,state['params'])
        assert self.params == state['params'],err_m
        
    def loads(self,state):
        self.scores = state['scores']
        self.score = state['score']
        self.params = state['params']
        self.split_pred = state['split_pred']

        