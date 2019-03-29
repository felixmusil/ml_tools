from .KRR import KRR
from ..kernels.kernels import make_kernel
from ..base import np
from ..utils import return_deepcopy

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

        KNMp[:Nr] /= lambdas[0]
        Yp[:Nr] /= lambdas[0]
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