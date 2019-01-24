from .dvr_radial_basis import get_descriptor,density
from ..base import AtomicDescriptorBase
from ..base import np,sp
from scipy.sparse import lil_matrix,csr_matrix
import re
from string import Template
from ..utils import tqdm_cs
from .utils import get_chunks,get_frame_slices


# def get_Nsoap(spkitMax,nmax,lmax):
#     Nsoap = 0
#     for sp1 in spkitMax:
#         for sp2 in spkitMax:
#             if sp1 == sp2:
#                 Nsoap += nmax*(nmax+1)*(lmax+1) / 2
#             elif sp1 > sp2:
#                 Nsoap += nmax**2*(lmax+1)
#     return Nsoap

def get_Nsoap(spkitMax,nmax,lmax):
    Nsoap = 0
    nspecies = len(spkitMax)
    Nsoap = nspecies**2 * (nmax+1)**2*(lmax+1)
    return Nsoap


class RawSoapInternal(AtomicDescriptorBase):
    def __init__(self,global_species=None,nocenters=None,rc=None, nmax=None,return_unlinsoap=False,
                 lmax=None, awidth=None,fast_avg=False,is_sparse=False,disable_pbar=False,leave=True):

        self.soap_params = dict(rc=rc, nmax=nmax, lmax=lmax, awidth=awidth,
                                global_species=global_species,
                               nocenters=nocenters)
        self.fast_avg = fast_avg
        self.is_sparse = is_sparse
        self.disable_pbar = disable_pbar
        self.leave = leave
        self.return_unlinsoap = return_unlinsoap

    def fit(self,X):
        return self
    def get_params(self,deep=True):
        params = dict(is_sparse=self.is_sparse,disable_pbar=False,
                        return_unlinsoap=self.return_unlinsoap,
                        fast_avg=self.fast_avg,leave=self.leave)
        params.update(**self.soap_params)
        return params

    def transform(self,X):

        rc = self.soap_params['rc']
        # in the implementation n goes from 0 to nmax instead of nmax-1
        nmax = self.soap_params['nmax']-1
        lmax = self.soap_params['lmax']
        awidth = self.soap_params['awidth']
        global_species = self.soap_params['global_species']
        nocenters = self.soap_params['nocenters']

        frames = X
        slices,strides = get_frame_slices(frames,nocenters=nocenters,
                                            fast_avg=self.fast_avg )

        Nenv = strides[-1]

        Nsoap = get_Nsoap(global_species,nmax,lmax)

        Nframe = len(frames)

        brcut = rc + 3.0*awidth
        gdens = density(nmax, lmax, brcut, awidth)

        centers = []
        for sp in global_species:
            if sp not in nocenters:
                centers.append(sp)

        if self.is_sparse:
            soaps = lil_matrix((Nenv,Nsoap),dtype=np.float64)
        else:
            soaps = np.empty((Nenv,Nsoap))

        for iframe in tqdm_cs(range(Nframe),desc='RawSoap',leave=self.leave,disable=self.disable_pbar):
            soap = get_descriptor(centers,frames[iframe], global_species, nmax, lmax, rc, gdens)
            soap = np.vstack(soap)

            soap /=  np.linalg.norm(soap,axis=1).reshape((-1,1))
            if self.fast_avg:
                soap = np.mean(soap,axis=0)

            if self.is_sparse:
                soap[np.abs(soap)<1e-13] = 0
                soaps[slices[iframe]] = lil_matrix(soap)
            else:
                soaps[slices[iframe]] = soap
        if self.is_sparse:
            soaps = soaps.tocsr(copy=False)
        self.slices = slices
        self.strides = strides

        if self.return_unlinsoap is True:
            nspecies = len(global_species)-len(nocenters)
            soaps = soaps.reshape((Nenv,nspecies,nspecies,nmax+1,nmax+1,lmax+1))

        return soaps

    def pack(self):
        state = dict(soap_params=self.soap_params,is_sparse=self.is_sparse,
                     slices=self.slices,strides=self.strides,
                     disable_pbar=self.disable_pbar)
        return state
    def unpack(self,state):
        err_m = 'soap_params are not consistent {} != {}'.format(self.soap_params,state['soap_params'])
        assert self.soap_params == state['soap_params'], err_m
        err_m = 'is_sparse are not consistent {} != {}'.format(self.is_sparse,state['is_sparse'])
        assert self.is_sparse == state['is_sparse'], err_m
        self.strides = state['strides']
        self.slices = state['slices']
    def loads(self,state):
        self.soap_params = state['soap_params']
        self.is_sparse = state['is_sparse']
        self.strides = state['strides']
        self.slices = state['slices']
        self.disable_pbar = state['disable_pbar']