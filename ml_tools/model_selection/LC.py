from builtins import range
from ..split import LCSplit,ShuffleSplit
from ..utils import tqdm_cs,return_deepcopy,score_func
from ..base import np,sp,BaseEstimator
import matplotlib.pyplot as plt

class LearningCurve(BaseEstimator):
    def __init__(self,trainer,model_params, n_repeats,train_sizes,test_size, seed=10):
        self.args = dict(n_repeats=n_repeats,train_sizes=train_sizes,test_size=test_size,seed=seed)
        self.model_params = model_params
        self.lc = LCSplit(ShuffleSplit, n_repeats=n_repeats,train_sizes=train_sizes,test_size=test_size, random_state=seed)
        self.trainer = trainer
        self.score_func = score_func

    def fit(self,X,y):
        self.trainer.is_precomputed = False
        self.scores = {k:[] for k in self.score_func}
        self.predictions = []

        for train,test in self.lc.split(y[:,None]):
            model = self.trainer.fit(train_ids=train,y_train=y,X_train=X,**self.model_params)

            K_test,y_true = self.trainer.prepare_kernel_and_targets(train_ids=train,test_ids=test,**self.model_params)
            y_pred = model.predict(K_test)
            self.predictions.append(dict(y_ref=y_true,y_pred=y_pred))
            for k,func in self.score_func.items():
                self.scores[k].append(func(y_true,y_pred))

        strides = np.cumsum([0]+list(self.args['n_repeats']))
        train_sizes = np.array(self.args['train_sizes'])
        self.score = [[],]*len(train_sizes)
        for itrain in range(len(train_sizes)):
            sc = {'Ntrain':train_sizes[itrain]}
            for k in self.scores:
                st = strides[itrain]
                nd = strides[itrain+1]
                sc[k] = np.mean(self.scores[k][st:nd])
            self.score[itrain] = sc

    def predict(self,X):
        raise 'LC does not predict'

    def plot(self,measure):
        Ntrains = [sc['Ntrain']  for sc in self.score]
        scores = [sc[measure]  for sc in self.score]
        ax = plt.gca()
        p = plt.plot(Ntrains,scores,'o')
        ax.set_yscale('log')
        ax.set_xscale('log')

        return ax
