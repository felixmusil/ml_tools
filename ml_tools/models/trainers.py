from .KRR import KRR,CommiteeKRR
from ..kernels.kernels import make_kernel
from ..base import np,TrainerBase
from ..utils import return_deepcopy
from ..split import ShuffleSplit,KFold
try:
    from collections.abc import Iterable
except:
    from collections import Iterable


class FullCovarianceTrainer(TrainerBase):
    def __init__(self, model_name='krr', kernel_name='gap', has_global_targets=True,self_contribution=None,feature_transformations=None,is_precomputed=False,has_forces=False, **kwargs):
        super(FullCovarianceTrainer,self).__init__(feature_transformations,has_global_targets)
        if has_forces is True:
            raise NotImplementedError()
        self.kwargs = kwargs
        self.kernel = make_kernel(kernel_name,**kwargs)
        self.kernel_name = kernel_name
        self.model_name = model_name

        self.self_contribution = self_contribution

        self.is_precomputed = is_precomputed
        self.has_forces = has_forces

        self.Y = None
        self.Y0 = None
        self.K = None
        self.Nr = None
        self.X_train = None
        self.Natoms = None

    def get_subset(self,ids=None,test_ids=None):
        if ids is None:
            return self.K, self.Y, self.Nr
        elif ids is not None and test_ids is None:
            return self.K[np.ix_(ids,ids)], self.Y[ids], len(ids)
        elif ids is not None and test_ids is not None:
            return self.K[np.ix_(test_ids,ids)], self.Y[test_ids], len(test_ids)
        else:
            raise RuntimeError()

    @return_deepcopy
    def get_params(self,deep=True):
        params = super(FullCovarianceTrainer,self).get_params()
        params.update(model_name=self.model_name,
          kernel_name=self.kernel_name,
          is_precomputed=self.is_precomputed,
          has_forces=self.has_forces,
          has_global_targets=self.has_global_targets,
          self_contribution=self.self_contribution)
        params.update(**self.kwargs)
        return params

    @return_deepcopy
    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(
            Y = self.Y,
            Y0 = self.Y0,
            Nr = self.Nr,
            K = self.K,
            X_train = self.X_train,
            Natoms = self.Natoms,
        )
        return state

    def loads(self,data):
        self.Y = data['Y']
        self.Y0 = data['Y0']
        self.Natoms = data['Natoms']
        self.K = data['K']
        self.Nr = data['Nr']
        self.X_train = data['X_train']

    def precompute(self, y_train, X_train, f_train=None, y_train_nograd=None, X_train_nograd=None):
        if f_train is not None or X_train_nograd is not None:
            raise NotImplementedError()
        is_global_target = dict(is_global=self.has_global_targets)
        kernel = self.kernel

        Nr1 = X_train.get_nb_sample(**is_global_target)
        Nr2 = 0
        if X_train_nograd is not None:
            Nr2 = X_train_nograd.get_nb_sample(**is_global_target)
            y_train = np.concatenate([y_train,y_train_nograd])

        Nr = Nr1 + Nr2
        Natoms = np.zeros(Nr)
        for iframe,sp in X_train.get_ids(**is_global_target):
            Natoms[iframe] += 1
        if X_train_nograd is not None:
            for iframe,sp in X_train_nograd.get_ids(**is_global_target):
                Natoms[Nr1+iframe] += 1

        if self.self_contribution is None:
            self.self_contribution = self.make_self_contribution(y_train,X_train,X_train_nograd,Natoms)

        Y0 = np.zeros(Nr)

        for iframe,sp in X_train.get_ids(**is_global_target):
            Y0[iframe] += self.self_contribution[sp]
        if X_train_nograd is not None:
            for iframe,sp in X_train_nograd.get_ids(**is_global_target):
                Y0[Nr1+iframe] += self.self_contribution[sp]

        if f_train is None:
            Ng = 0
        else:
            Ng = X_train.get_nb_sample(gradients=True)
            f = -f_train.reshape((-1,))
            self.has_forces = True

        N = Nr + Ng * 3

        Y = np.zeros((N,))

        Y[:Nr] = (y_train - Y0) # / Natoms

        if self.has_forces is True:
            Y[Nr:] = f

        K = np.zeros((N,N))
        K[:Nr1,:Nr1] = kernel.transform(X_train)

        if X_train_nograd is not None:
            K[Nr1:Nr,Nr1:Nr] = kernel.transform(X_train_nograd,eval_gradient=(False,False))
            ee = kernel.transform(X_train,X_train_nograd,eval_gradient=(False,False))
            K[:Nr1,Nr1:Nr] = ee
            K[Nr1:Nr,:Nr1] = ee.T

        if f_train is not None:
            K[Nr:,Nr:] = kernel.transform(X_train,eval_gradient=(True,True))
            # some other stuff need to be written

        self.Y = Y
        self.Y0 = Y0
        self.Natoms = Natoms
        self.K = K

        self.Nr = Nr
        self.X_train = X_train

        self.is_precomputed = True

    def prepare_kernel_and_targets(self,lambdas=[1e-2], jitter=1e-9, train_ids=None,test_ids=None, y_train=None, X_train=None, f_train=None, y_train_nograd=None, X_train_nograd=None):
        if self.is_precomputed is False:
            self.precompute(y_train, X_train, f_train, y_train_nograd, X_train_nograd)

        if self.has_forces is True:
            raise NotImplementedError()
        K,Y,Nr =  self.get_subset(train_ids,test_ids)

        delta = np.std(Y[:Nr]) / np.mean(K.diagonal()[:Nr])

        K = K.copy()
        Y = Y.copy()

        if test_ids is None:
            # jitter for numerical stability
            K[np.diag_indices_from(K[:Nr,:Nr])] += lambdas[0]**2 / delta **2 + jitter

            if f_train is not None:
                raise NotImplementedError()
                K[np.diag_indices_from(K[:Nr,:Nr])] += lambdas[1]**2 / delta **2 + jitter

            # self.K = K
            self.delta = delta

        return K,Y

    def fit(self, lambdas=[1e-2], jitter=1e-9, train_ids=None, y_train=None, X_train=None, f_train=None, y_train_nograd=None, X_train_nograd=None):

        K,Y = self.prepare_kernel_and_targets(lambdas=lambdas,jitter=jitter,train_ids=train_ids,y_train=y_train,X_train=X_train,f_train=f_train,y_train_nograd=y_train_nograd,X_train_nograd=X_train_nograd)

        if self.model_name == 'krr':
            model_params = dict(kernel=self.kernel, X_train=self.X_train, feature_transformations=self.feature_transformations,has_global_targets=self.has_global_targets,self_contribution=self.self_contribution)

            model = KRR(**model_params)


        model.fit(K,Y)

        return model



class SoRTrainer(TrainerBase):
    def __init__(self, model_name='krr', kernel_name='gap',has_global_targets=True,self_contribution=None,feature_transformations=None,is_precomputed=False,has_forces=False, **kwargs):
        super(SoRTrainer,self).__init__(feature_transformations,has_global_targets)
        self.is_precomputed = is_precomputed
        self.kwargs = kwargs
        self.kernel = make_kernel(kernel_name,**kwargs)
        self.model_name = model_name
        self.has_forces = has_forces
        self.self_contribution = self_contribution

        self.Y = None
        self.Y0 = None
        self.KMM = None
        self.KNM = None
        self.Nr = None
        self.X_pseudo = None
        self.Natoms = None

    def get_subset(self,ids=None,test_ids=None):
        if ids is None:
            return self.KNM,self.Y, self.KMM,self.Nr,self.Natoms
        else:
            return self.KNM[ids],self.Y[ids], self.KMM, len(ids),self.Natoms[ids]

    @return_deepcopy
    def get_params(self,deep=True):
        params = super(SoRTrainer,self).get_params()
        params.update(model_name=self.model_name,
          kernel_name=self.kernel_name,
          is_precomputed=self.is_precomputed,
          has_forces=self.has_forces,
          self_contribution=self.self_contribution,)
        params.update(**self.kwargs)
        return params

    @return_deepcopy
    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(
            Y = self.Y,
            Y0 = self.Y0,
            KMM = self.KMM,
            KNM = self.KNM,
            Nr = self.Nr,
            X_pseudo = self.X_pseudo,
            Natoms = self.Natoms,
        )
        return state

    def loads(self,data):
        self.Y = data['Y']
        self.Y0 = data['Y0']
        self.Natoms = data['Natoms']
        self.KMM = data['KMM']
        self.KNM = data['KNM']
        self.Nr = data['Nr']
        self.X_pseudo = data['X_pseudo']

    def precompute(self, y_train, X_train, X_pseudo, f_train=None, y_train_nograd=None, X_train_nograd=None):
        kernel = self.kernel
        is_global_target = dict(is_global=self.has_global_targets)

        M = X_pseudo.get_nb_sample()
        Nr1 = X_train.get_nb_sample(**is_global_target)
        Nr2 = 0
        if X_train_nograd is not None:
            Nr2 = X_train_nograd.get_nb_sample(**is_global_target)
            y_train = np.concatenate([y_train,y_train_nograd])

        Nr = Nr1 + Nr2
        Natoms = np.zeros(Nr)
        for iframe,sp in X_train.get_ids(**is_global_target):
            Natoms[iframe] += 1
        if X_train_nograd is not None:
            for iframe,sp in X_train_nograd.get_ids(**is_global_target):
                Natoms[Nr1+iframe] += 1

        if self.self_contribution is None:
            self.self_contribution = self.make_self_contribution(y_train,X_train,X_train_nograd,Natoms)

        Y0 = np.zeros(Nr)

        for iframe,sp in X_train.get_ids(**is_global_target):
            Y0[iframe] += self.self_contribution[sp]
        if X_train_nograd is not None:
            for iframe,sp in X_train_nograd.get_ids(**is_global_target):
                Y0[Nr1+iframe] += self.self_contribution[sp]

        if f_train is None:
            Ng = 0

        else:
            Ng = X_train.get_nb_sample(gradients=True)
            f = -f_train.reshape((-1,))
            self.has_forces = True

        N = Nr + Ng * 3

        Y = np.zeros((N,))
        # per atom formation energy
        Y[:Nr] = (y_train - Y0) # / Natoms
        # Y[:Nr] -= Y[:Nr].mean()

        if self.has_forces is True:
            Y[Nr:] = f

        KMM = np.zeros((M,M))
        KNM = np.zeros((N,M))


        KMM = kernel.transform(X_pseudo,X_pseudo)

        KNM[:Nr1] = kernel.transform(X_train,X_pseudo,eval_gradient=(False,False)) #  / np.sqrt(K_diag)

        if X_train_nograd is not None:
            KNM[Nr1:Nr] = kernel.transform(X_train_nograd,X_pseudo,eval_gradient=(False,False))
        if f_train is not None:
            KNM[Nr:] = kernel.transform(X_train,X_pseudo,eval_gradient=(True,False))

        self.Y = Y
        self.Y0 = Y0
        self.Natoms = Natoms
        self.KMM = KMM
        self.KNM = KNM

        self.Nr = Nr
        self.X_pseudo = X_pseudo

        self.is_precomputed = True

    def prepare_kernel_and_targets(self,lambdas, jitter=1e-8, train_ids=None, test_ids=None,y_train=None, X_train=None, X_pseudo=None, f_train=None, y_train_nograd=None, X_train_nograd=None):
        if test_ids is not None:
            train_ids = test_ids
        if self.is_precomputed is False:
            self.precompute(y_train, X_train, X_pseudo, f_train, y_train_nograd, X_train_nograd)
        KNM,Y,KMM,Nr,Natoms = self.get_subset(train_ids)

        delta = np.std(Y[:Nr])

        KNMp = KNM.copy()
        Yp = Y.copy()
        KMMp = KMM.copy()
        if test_ids is None:
            # lambdas[0] is provided per atom
            KNMp[:Nr] /=  lambdas[0] / delta * np.sqrt(Natoms)[:,None]
            Yp[:Nr] /= lambdas[0] / delta * np.sqrt(Natoms)
            # jitter for numerical stability
            KMMp[np.diag_indices_from(KMMp)] += jitter
            if self.has_forces is True:
                KNMp[Nr:] /= lambdas[1] / delta
                Yp[Nr:] /= lambdas[1] / delta
            Y = np.dot(KNMp.T,Yp)

            K = KMMp + np.dot(KNMp.T,KNMp)
        else:
            K = KNMp

        self.K = K
        self.delta = delta

        return K,Y

    def fit(self, lambdas, jitter=1e-8, train_ids=None, y_train=None, X_train=None, X_pseudo=None, f_train=None, y_train_nograd=None, X_train_nograd=None):
        K,Y = self.prepare_kernel_and_targets(lambdas=lambdas,jitter=jitter,train_ids=train_ids,y_train=y_train,X_train=X_train,X_pseudo=X_pseudo,f_train=f_train,y_train_nograd=y_train_nograd,X_train_nograd=X_train_nograd)

        if self.model_name == 'krr':
            model_params = dict(kernel=self.kernel, X_train=self.X_pseudo, feature_transformations=self.feature_transformations,has_global_targets=self.has_global_targets,self_contribution=self.self_contribution)

            model = KRR(**model_params)

        model.fit(K,Y)

        return model


class CommiteeSoRTrainer(SoRTrainer):
    def __init__(self,n_models, sampling_method='resample 0.66', seed=10,
    prediction_repeat_threshold=None,model_name='krr', kernel_name='gap', self_contribution=None,representation=None,is_precomputed=False,has_forces=False, **kwargs):
        super(CommiteeSoRTrainer,self).__init__(model_name, kernel_name, self_contribution,representation,is_precomputed,has_forces, **kwargs)
        self.sampling_method = sampling_method
        self.n_models = n_models
        self.seed = seed
        if prediction_repeat_threshold is None:
            self.prediction_repeat_threshold = self._get_resampling_repeat_threshold()
        else:
            self.prediction_repeat_threshold = prediction_repeat_threshold

    def _get_resampling_repeat_threshold(self):
        n_models = self.n_models
        probs = np.zeros(n_models)
        for ii in range(n_models):
            probs[ii] = self._probability_not_having_missing_predictions(ii,n_models,0.666)
        probs = np.abs(probs - 0.1)
        k = np.argmin(probs)
        return k

    @staticmethod
    def _probability_not_having_missing_predictions(k=4, n=16, p=0.66):
        """Gives the probability of having at most k times true event in n trials where each trials has probability p of being true.
        Here want to have a good probability of having more than k predictions for each of the samples given n resampled models. """
        from scipy.special import bdtr
        return bdtr(k, n, p)

    def fit(self,Lambda, jitter, y_train=None, X_train=None, X_pseudo=None):
        if self.is_precomputed is False:
            self.precompute(y_train, X_train, X_pseudo)

        if self.sampling_method == 'resample 0.66':
            train_size = int(0.66*n_train)
            splitter = ShuffleSplit(n_splits=self.n_models,
                                train_size=train_size,
                                random_state=self.seed)

        elif self.sampling_method == '2-fold':
            splitter = KFold(n_splits=2,shuffle=True,random_state=self.seed)

        if self.model_name == 'krr':
            model = CommiteeKRR(self.jitter, self.kernel, self.X_pseudo, self.representation,self_contribution=aa)

            k = self.prediction_repeat_threshold
            model = self.fit_krr(model,splitter,k)

        return model

    def fit_krr(self,model,splitter,prediction_repeat_threshold):
        alphas = []
        n_train = len(self.Y)
        # counter for the number of times a prediction has been made for the structure at hand
        rsny = np.zeros(n_train,int)
        # predictions made for the structure at hand
        rsy = np.zeros(n_train)
        # square of the predictions made for the structure at hand
        rsy2 = np.zeros(n_train)

        for train,test in splitter.split(self.Y):
            KNMp = self.KNM[train]
            Yp = self.Y[train]
            KNMp /= Lambda
            Yp /= Lambda
            K = self.KMM + np.dot(KNMp.T,KNMp)
            Y = np.dot(KNMp.T,Yp)
            model.fit(K,Y)

            Ktest = self.KNM[test]
            internal_true.append(self.Y[test])
            ypred = model.predict(Ktest)
            rsy[test] += ypred
            rsy2[test] += ypred**2
            rsny[test] += 1

        # mask to select those strucures for which at least 4 predictions exist
        selstat = np.where(rsny>prediction_repeat_threshold)[0]
        # mean predictions over models for strucures for which at least 4 predictions exist
        ybest = rsy[selstat]/rsny[selstat]
        # standard deviations of predictions over models for strucures for which at least 4 predictions exist
        yerr = np.sqrt(rsy2[selstat]/rsny[selstat] - (rsy[selstat]/rsny[selstat])**2)

        # measure of the quality of
        subsampling_correction = np.sqrt(np.mean((ybest - ytrain)**2/yerr**2))

        model.subsampling_correction = subsampling_correction

        return model

    @return_deepcopy
    def get_params(self,deep=True):
        aa = super(CommiteeSoRTrainer,self).get_params()
        aa['n_models'] = n_models
        aa['sampling_method'] = sampling_method
        aa['seed'] = seed
        aa['prediction_repeat_threshold'] = prediction_repeat_threshold
        return aa

    @return_deepcopy
    def dumps(self):
        state = super(CommiteeSoRTrainer,self).dumps()
        state['init_params'] = self.get_params()
        return state

    def loads(self,data):
        super(CommiteeSoRTrainer,self).loads(data)
