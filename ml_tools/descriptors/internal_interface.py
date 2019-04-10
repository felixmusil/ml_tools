from .dvr_radial_basis import get_descriptor,density
from ..base import AtomicDescriptorBase
from ..base import np,sp
from scipy.sparse import lil_matrix,csr_matrix
import re
from string import Template
from ..utils import tqdm_cs,return_deepcopy
from .utils import get_chunks,get_frame_slices,get_frame_neigbourlist
from .feature import Representation




class RawSoapInternal(AtomicDescriptorBase):
    def __init__(self,global_species=None,nocenters=None,rc=None, nmax=None,return_unlinsoap=False, lmax=None, awidth=None,fast_avg=False,is_sparse=False,disable_pbar=False,leave=False):

        self.soap_params = dict(
            rc=float(rc), nmax=int(nmax), lmax=int(lmax),awidth=float(awidth),
            global_species=list(global_species),nocenters=list(nocenters))
        self.fast_avg = fast_avg
        self.is_sparse = is_sparse
        self.disable_pbar = disable_pbar
        self.leave = leave
        # self.return_unlinsoap = return_unlinsoap

        brcut = rc + 3.0*awidth
        # in the implementation n goes from 0 to nmax instead of nmax-1
        self.gdens = density(nmax-1, lmax, brcut, awidth)

    @return_deepcopy
    def get_params(self,deep=True):
        params = dict(is_sparse=self.is_sparse,disable_pbar=False,
                        # return_unlinsoap=self.return_unlinsoap,
                        fast_avg=self.fast_avg,leave=self.leave)
        params.update(**self.soap_params)
        return params
    @return_deepcopy
    def dumps(self):
        state = {}
        state['init_params'] = self.get_params()
        state['data'] = dict(
        )
        return state

    def loads(self,data):
        pass

    def fit(self,X):
        return self

    @staticmethod
    def get_Nsoap(spkitMax,nmax,lmax):
        Nsoap = 0
        nspecies = len(spkitMax)
        Nsoap = nspecies**2 * (nmax+1)**2*(lmax+1)
        return Nsoap

    def init_data(self, frames):
        soap_params = self.soap_params
        chunk_len = 100
        fast_avg = self.fast_avg

        global_species = soap_params['global_species']
        nocenters = soap_params['nocenters']
        rc = soap_params['rc']
        awidth = soap_params['awidth']
        # in the implementation n goes from 0 to nmax instead of nmax-1
        nmax = soap_params['nmax'] - 1
        lmax = soap_params['lmax']

        self.atomic_types = [[]]*len(frames)
        for iframe, frame in enumerate(frames):
            numbers = frame.get_atomic_numbers()
            self.atomic_types[iframe] = np.setdiff1d(numbers,nocenters)

        Nfeature = self.get_Nsoap(global_species,nmax,lmax)

        Ncenter,self.slices,strides = get_frame_slices(frames,nocenters=nocenters, fast_avg=fast_avg)

        with_gradients = False
        if with_gradients is False:
            self.slices_gradients = [None] * len(frames)
            features = Representation(Ncenter=Ncenter, Nfeature=Nfeature, chunk_len=chunk_len,hyperparams=soap_params)


        elif with_gradients is True:
            Nneighbour,strides_gradients,self.slices_gradients = get_frame_neigbourlist(frames,nocenters=nocenters)

            features = Representation(Ncenter=Ncenter, Nfeature=Nfeature,  Nneighbour=Nneighbour, has_gradients=True,chunk_len=chunk_len,hyperparams=soap_params)

        return features

    def insert_to(self, features, iframe, results):
        rep = results['rep']

        desc_mapping = np.zeros((rep.shape[0],1))

        species = self.atomic_types[iframe]

        features.insert(
                self.slices[iframe], rep, species, desc_mapping
        )

    def transform(self,X):

        rc = self.soap_params['rc']
        # in the implementation n goes from 0 to nmax instead of nmax-1
        nmax = self.soap_params['nmax']-1
        lmax = self.soap_params['lmax']
        awidth = self.soap_params['awidth']
        global_species = self.soap_params['global_species']
        nocenters = self.soap_params['nocenters']

        frames = X
        Nsoap = self.get_Nsoap(global_species,nmax,lmax)
        Nenv,slices,strides = get_frame_slices(frames,nocenters=nocenters,
                                            fast_avg=self.fast_avg )

        Nframe = len(frames)

        centers = []
        for sp in global_species:
            if sp not in nocenters:
                centers.append(sp)

        if self.is_sparse:
            soaps = lil_matrix((Nenv,Nsoap),dtype=np.float64)
        else:
            soaps = self.init_data(frames)

        for iframe in tqdm_cs(range(Nframe),desc='RawSoap',leave=self.leave,disable=self.disable_pbar):
            soap = get_descriptor(centers,frames[iframe], global_species, nmax, lmax, rc, self.gdens)
            # soap = np.vstack(soap)

            soap /=  np.linalg.norm(soap,axis=1).reshape((-1,1))
            if self.fast_avg:
                soap = np.mean(soap,axis=0)

            if self.is_sparse:
                soap[np.abs(soap)<1e-13] = 0
                soaps[self.slices[iframe]] = lil_matrix(soap)
            else:
                self.insert_to(soaps,iframe,dict(rep=soap))

        if self.is_sparse:
            soaps = soaps.tocsr(copy=False)

        self.strides = strides

        # if self.return_unlinsoap is True:
        #     nspecies = len(global_species)-len(nocenters)
        #     soaps = soaps.reshape((Nenv,nspecies,nspecies,nmax+1,nmax+1,lmax+1))

        return soaps
