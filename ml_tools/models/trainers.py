from .KRR import KRR,CommiteeKRR
from ..kernels.kernels import make_kernel
from ..base import np
from ..utils import return_deepcopy
from ..split import ShuffleSplit,KFold

class TrainerSoR(object):
    def __init__(self, model_name='krr', kernel_name='gap', self_energies=None,representation=None,is_precomputed=False,has_forces=False, **kwargs):
        self.is_precomputed = is_precomputed
        self.kwargs = kwargs
        self.kernel = make_kernel(kernel_name,**kwargs)
        self.model_name = model_name
        self.has_forces = has_forces
        self.representation = representation
        self.representation.disable_pbar = True
        self.self_energies = self_energies

        self.Y = None
        self.Y0 = None
        self.KMM = None
        self.KNM = None
        self.Nr = None
        self.X_pseudo = None
        self.Natoms = None
    @return_deepcopy
    def get_params(self,deep=True):
        return dict(
          model_name=self.model_name,
          kernel_name=self.kernel_name,
          representation=self.representation,
          is_precomputed=self.is_precomputed,
          has_forces=self.has_forces,
          kwargs=self.kwargs,
          self_energies=self.self_energies,
        )

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
        self.self_energies = data['self_energies']

    def precompute(self, y_train, X_train, X_pseudo, f_train=None, y_train_nograd=None, X_train_nograd=None):
        kernel = self.kernel

        M = X_pseudo.get_nb_sample()
        Nr1 = X_train.get_nb_sample()
        Nr2 = 0
        if X_train_nograd is not None:
            Nr2 = X_train_nograd.get_nb_sample()
            y_train = np.concatenate([y_train,y_train_nograd])

        Nr = Nr1 + Nr2

        if self.self_energies is None:
            Y0 = y_train.mean()
        else:
            Y0 = np.zeros(Nr)
            Natoms = np.zeros(Nr)
            for iframe,sp in X_train.get_ids():
                Y0[iframe] += self.self_energies[sp]
                Natoms[iframe] += 1
            if X_train_nograd is not None:
                for iframe,sp in X_train_nograd.get_ids():
                    Y0[Nr1+iframe] += self.self_energies[sp]
                    Natoms[Nr1+iframe] += 1
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

    def fit(self, lambdas, jitter, y_train=None, X_train=None, X_pseudo=None, f_train=None, y_train_nograd=None, X_train_nograd=None):
        if self.is_precomputed is False:
            self.precompute(y_train, X_train, X_pseudo, f_train, y_train_nograd, X_train_nograd)
        Nr = self.Nr
        KNMp = self.KNM.copy()
        Yp = self.Y.copy()

        # lambdas[0] is provided per atom
        KNMp[:Nr] /= lambdas[0] * np.sqrt(self.Natoms).reshape((-1,1))
        Yp[:Nr] /= lambdas[0] * np.sqrt(self.Natoms)
        if self.has_forces is True:
          KNMp[Nr:] /= lambdas[1]
          Yp[Nr:] /= lambdas[1]

        if self.model_name == 'krr':
            if self.self_energies is None:
                aa = self.Y0
            else:
                aa = self.self_energies
            model = KRR(jitter, self.kernel, self.X_pseudo, self.representation,aa)
            K = self.KMM + np.dot(KNMp.T,KNMp)
            Y = np.dot(KNMp.T,Yp)
            model.fit(K,Y)
            self.K = K

        return model


class TrainerCommiteeSoR(TrainerSoR):
    def __init__(self,n_models, sampling_method='resample 0.66', seed=10,
    prediction_repeat_threshold=None,model_name='krr', kernel_name='gap', self_energies=None,representation=None,is_precomputed=False,has_forces=False, **kwargs):
        super(TrainerCommiteeSoR,self).__init__(model_name, kernel_name, self_energies,representation,is_precomputed,has_forces, **kwargs)
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
            if self.self_energies is None:
                aa = self.Y0
            else:
                aa = self.self_energies

            model = CommiteeKRR(self.jitter, self.kernel, self.X_pseudo, self.representation,self_energies=aa)

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
        aa = super(TrainerCommiteeSoR,self).get_params()
        aa['n_models'] = n_models
        aa['sampling_method'] = sampling_method
        aa['seed'] = seed
        aa['prediction_repeat_threshold'] = prediction_repeat_threshold
        return aa

    @return_deepcopy
    def dumps(self):
        state = super(TrainerCommiteeSoR,self).dumps()
        state['init_params'] = self.get_params()
        return state

    def loads(self,data):
        super(TrainerCommiteeSoR,self).loads(data)
