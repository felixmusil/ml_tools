from ..base import BaseEstimator,TransformerMixin,FeatureBase,np
from ..utils import tqdm_cs,return_deepcopy
from scipy.sparse.linalg import  svds

class CURFilter(BaseEstimator,TransformerMixin):
    def __init__(self,Nselect,act_on='sample',is_deterministic=True,seed=10):
        self.Nselect = Nselect
        if act_on in ['sample','sample per specie','feature']:
            self.act_on = act_on
        else:
            raise 'Wrong input: {}'.format(act_on)
        self.is_deterministic = is_deterministic
        self.seed = seed
        self.trained = False
        self.selected_ids = None

    @return_deepcopy
    def get_params(self,deep=True):
        params = dict(Nselect=self.Nselect,act_on=self.act_on,is_deterministic=self.is_deterministic,
                         seed=self.seed)
        return params
        
    @return_deepcopy
    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(selected_ids=self.selected_ids,
                            trained=self.trained)
        return state

    def loads(self, data):
        if isinstance(data['selected_ids'],dict):
            self.selected_ids = {int(k):v for k,v in data['selected_ids'].items()}
        else:
            self.selected_ids = data['selected_ids']
        self.trained = data['trained']

    def fit(self,X,gradients=False):
        if self.act_on in ['sample per specie']:
            self.selected_ids = {}
            x = {}
            if isinstance(X,dict):
                x[1] = X['feature_matrix']
            elif isinstance(X,FeatureBase):
                usp = np.unique(X.get_species())
                for sp in usp:
                    x[sp] = X.get_data(specie=sp)
                # if gradients is False:
                #     x = X.get_data()
                # elif gradients is True:
                #     x = X.get_data(gradients=True)
            else:
                x[1] = X

            for sp in x:
                print sp
                self.selected_ids[sp] = do_CUR(x[sp],self.Nselect[sp],self.act_on,self.is_deterministic,self.seed)
            self.trained = True
        return self

    def transform(self,X):
        if self.act_on == 'sample' and self.trained is True:
            # Only the training set needs to be sparsified
            # at prediction time it should do nothing to the
            # new samples
            self.trained = False
            if isinstance(X,FeatureBase):
                Xr = X.extract_pseudo_input(self.selected_ids[:self.Nselect])
            else:
                Xr = X[self.selected_ids[:self.Nselect],:]
        elif self.act_on == 'sample per specie':
            self.trained = False
            if isinstance(X,FeatureBase):
                selected_ids = {}
                for sp in self.selected_ids:
                    selected_ids[sp] = self.selected_ids[sp][:self.Nselect[sp]]
                Xr = X.extract_pseudo_input(selected_ids)
            else:
                Xr = X[self.selected_ids[:self.Nselect],:]
        elif self.act_on == 'feature':
            if isinstance(X,FeatureBase):
                Xr = X.extract_feature_selection(self.selected_ids[:self.Nselect])
            else:
                Xr = X[:,self.selected_ids[:self.Nselect]]

        return Xr



def do_CUR(X, Nsel, act_on='sample', is_deterministic=False,seed=10,verbose=True):
    """ Apply CUR selection of Nsel rows or columns of the
    given covariance matrix. """

    U,S,VT = svds(X, Nsel)
    if 'sample' in act_on:
        weights = np.mean(np.square(U),axis=1)
    elif 'feature' in act_on:
        weights = np.mean(np.square(VT),axis=0)

    if is_deterministic is True:
        # sorting is smallest to largest hence the minus
        sel = np.argsort(-weights)[:Nsel]

    elif is_deterministic is False:
        np.random.seed(seed)
        # sorting is smallest to largest hence the minus
        sel = np.argsort(np.random.rand(*weights.shape) - weights)[:Nsel]

    if verbose is True:
        if 'sample' in act_on:
            C = X[sel,:]
        elif 'feature' in act_on:
            C = X[:,sel]
        Cp = np.linalg.pinv(C)
        err = np.sum((X - np.dot(np.dot(X,Cp),C))**2)
        print('Err={:.3e}'.format(err))

    return sel