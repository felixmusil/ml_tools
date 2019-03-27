
from ..base import KernelBase,FeatureBase
from ..base import np,sp
from ..math_utils import power,average_kernel
from ..math_utils.power_kernel import (sum_power_no_species,derivative_sum_power_no_species,sum_power_diff_species,derivative_sum_power_diff_species,sum_power_no_species_self,
sum_power_diff_species_self)
from ..utils import tqdm_cs,is_autograd_instance
#from ..math_utils.basic import power,average_kernel
from scipy.sparse import issparse
# from collections.abc import Iterable


registered_kernel = ['power', 'power_sum', 'gap']

def make_kernel(name, **kwargs):
    if name == registered_kernel[0]:
        kernel = KernelPower(**kargs[0])
    elif name == registered_kernel[1]:
        kernel = KernelSum(kernel_type = 'power', **kwargs)
    elif name == registered_kernel[2]:
        kernel = KernelSum(kernel_type = 'power_gap', **kwargs)

    return kernel



class KernelPower(KernelBase):
    def __init__(self,zeta):
        self.zeta = zeta
    def fit(self,X):
        return self
    def get_params(self,deep=True):
        params = dict(zeta=self.zeta)
        return params
    def set_params(self,**params):
        self.zeta = params['zeta']
    def transform(self,X,X_train=None):
        return self(X,Y=X_train)

    def __call__(self, X, Y=None,zeta=None, eval_gradient=(False,False)):
        # X should be shape=(Nsample,Mfeature)
        # if not assumes additional dims are features

        if zeta is None:
            zeta = self.zeta
        if isinstance(X, FeatureBase):
            Xi = X.get_data(gradients=eval_gradient[0])
        elif len(X.shape) > 2:
            Nenv = X.shape[0]
            Xi = X.reshape((Nenv,-1))
        else:
            Xi = X

        if isinstance(Y, FeatureBase):
            Yi = Y.get_data(gradients=False)
        elif Y is None:
            Yi = Xi
        elif len(Y.shape) > 2:
            Nenv = Y.shape[0]
            Yi = Y.reshape((Nenv,-1))
        else:
            Yi = Y

        if issparse(Xi) is False:
            if eval_gradient[0] is True:
                return self.dK_dr(X,Y)
            elif eval_gradient[0] is False:
                return power(np.dot(Xi,Yi.T),zeta)

        elif issparse(Xi) is True:
            N, M = Xi.shape[0],Yi.shape[0]
            kk = np.zeros((N,M))
            Xi.dot(Yi.T).todense(out=kk)
            return power(kk,zeta)

    def dK_dr(self,X,Y):
        zeta = self.zeta
        dXi_dr = X.get_data(gradients=True)
        Yi = Y.get_data(gradients=False)
        if zeta == 1:
            return np.dot(dXi_dr,Yi.T)
        else:
            mapping = X.get_mapping(gradients=True)
            Xi = X.get_data(gradients=False)
            Xi = Xi[mapping]
            k = zeta * self(Xi,Yi,zeta=zeta-1)

            k = np.concatenate([k,k,k],axis=0)

            return np.einsum('jm,jd,md->jm',k,dXi_dr,Yi,optimize='optimal')

    def K_diag(self,Xfeat,zeta=None,eval_gradient=False):

        if isinstance(Xfeat, FeatureBase):
            N = Xfeat.get_nb_environmental_elements(gradients=eval_gradient)

            K_diag = np.ones((N,1))
            Xfeat.set_chunk_len(1)

            rep = Xfeat.get_data(gradients=eval_gradient)
            for it in range(N):
                K_diag[it] = self(rep[it],zeta)
        else:
            N = Xfeat.shape[0]
            K_diag = np.ones((N,1))
            for it in range(N):
                K_diag[it] = self(Xfeat[it],zeta) # power(np.dot(Xfeat[it],Xfeat[it]),zeta)

        return K_diag

    def pack(self):
        state = dict(zeta=self.zeta)
        return state
    def unpack(self,state):
        err_m = 'zetas are not consistent {} != {}'.format(self.zeta,state['zeta'])
        assert self.zeta == state['zeta'], err_m

    def loads(self,state):
        self.zeta = state['zeta']

class KernelSum(KernelBase):
    def __init__(self,kernel_type,**kwargs):
        self.kwargs = kwargs
        self.kernel_type = kernel_type
        self.set_kernel_func()

    def set_kernel_func(self):
        if self.kernel_type == 'power':
            self.k = sum_power_no_species
            self.k_self = sum_power_no_species_self
            self.dk_dr = derivative_sum_power_no_species
            self.zeta = self.kwargs['zeta']
        if self.kernel_type == 'power_gap':
            self.k = sum_power_diff_species
            self.k_self = sum_power_diff_species_self
            self.dk_dr = derivative_sum_power_diff_species
            self.zeta = self.kwargs['zeta']

    def fit(self,X):
        return self
    def get_params(self,deep=True):
        params = dict(kernel_type=self.kernel_type,kwargs=self.kwargs)
        return params
    def set_params(self,**params):
        self.kernel_type = params['kernel_type']
        self.kwargs = params['kwargs']

        self.set_kernel_func()

    def __call__(self, X, Y, is_square=False, eval_gradient=(False,False)):

        N = X.get_nb_sample(eval_gradient[0])
        M = Y.get_nb_sample(eval_gradient[1])

        Xi = X.get_data()
        Xids = X.get_ids()
        Yi = Y.get_data()
        Yids = Y.get_ids()

        if eval_gradient[0] is False:
            kernel = np.ones((N,M))
            if is_square is True:
                self.k_self(kernel,self.zeta,Xi,Xids)
            elif is_square is False:
                self.k(kernel,self.zeta,Xi,Xids,Yi,Yids)
        elif eval_gradient[0] is True:
            kernel = np.ones((N,3,M))
            dXi_dr = X.get_data(True)
            Xids = X.get_ids(True)
            self.dk_dr(kernel,self.zeta,Xi,dXi_dr,Xids,Yi,Yids)
            kernel = kernel.reshape(-1,M)

        return kernel

    def transform(self, X, X_train=None, eval_gradient=(False,False)):
        if isinstance(X,FeatureBase):
            Xfeat= X
        is_square = False
        if X_train is None:
            Yfeat = X
            is_square = True
        elif isinstance(X_train,FeatureBase):
            Yfeat = X_train
        K = self(Xfeat,Yfeat,is_square,eval_gradient)
        return K

    def K_diag(self,Xfeat,eval_gradient=False):

        return K_diag

    def dumps(self):
        state = self.get_params()
        return state
    def loads(self,state):
        self.set_params(state)



