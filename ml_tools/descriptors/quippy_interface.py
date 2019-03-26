from ..base import AtomicDescriptorBase,FeatureBase
from quippy import descriptors
from ..base import np,sp
from scipy.sparse import lil_matrix,csr_matrix
import re
from string import Template
from ..utils import tqdm_cs
from .utils import get_chunks,get_frame_slices,get_frame_neigbourlist
from .feature import DenseFeature,FeatureWithGrad

def ase2qp(aseatoms):
    from quippy import Atoms as qpAtoms
    positions = aseatoms.get_positions()
    cell = aseatoms.get_cell()
    numbers = aseatoms.get_atomic_numbers()
    pbc = aseatoms.get_pbc()
    return qpAtoms(numbers=numbers,cell=cell,positions=positions,pbc=pbc)

def get_Nsoap(spkitMax,nmax,lmax):
    Nsoap = 0
    for sp1 in spkitMax:
        for sp2 in spkitMax:
            if sp1 == sp2:
                Nsoap += nmax*(nmax+1)*(lmax+1) / 2
            elif sp1 > sp2:
                Nsoap += nmax**2*(lmax+1)
    return Nsoap + 1

def get_rawsoap(frame,soapstr,nocenters, global_species, rc, nmax, lmax,awidth,
                centerweight,cutoff_transition_width,
                 cutoff_dexp, cutoff_scale,cutoff_rate,grad=False):

    global_speciesstr = '{'+re.sub('[\[,\]]', '', str(global_species))+'}'
    nspecies = len(global_species)

    spInFrame = np.unique(frame.get_atomic_numbers())
    # makes sure that the nocenters is propely adapted to the species present in the frame
    nocenterInFrame = []
    for nocenter in nocenters:
        if nocenter in spInFrame:
            nocenterInFrame.append(nocenter)

    centers = []
    ncentres = len(spInFrame) - len(nocenterInFrame)
    for z in spInFrame:
        if z in nocenterInFrame:
            continue
        centers.append(str(z))
    centers = '{'+' '.join(centers)+'} '

    soapstr2 = soapstr.substitute(nspecies=nspecies, ncentres=ncentres,cutoff_rate=cutoff_rate,
                                  species=global_speciesstr, centres=centers,
                                  cutoff_transition_width=cutoff_transition_width,
                                 rc=rc, nmax=nmax, lmax=lmax, awidth=awidth,cutoff_dexp=cutoff_dexp,
                                  cutoff_scale=cutoff_scale,centerweight=centerweight)
    desc = descriptors.Descriptor(soapstr2)

    return desc.calc(frame, grad=grad)


class RawSoapQUIP(AtomicDescriptorBase):
    def __init__(self,global_species=None,nocenters=None,rc=None, nmax=None,
                 lmax=None, awidth=None,centerweight=None,cutoff_transition_width=None, cutoff_rate=None,
                 cutoff_dexp=None, cutoff_scale=None,fast_avg=False,is_sparse=False,disable_pbar=False, grad=False):
        if global_species is None and rc is None:
            pass
        self.soap_params = dict(rc=rc, nmax=nmax, lmax=lmax, awidth=awidth,cutoff_rate=cutoff_rate,grad=grad,
                                cutoff_transition_width=cutoff_transition_width,
                                cutoff_dexp=cutoff_dexp, cutoff_scale=cutoff_scale,
                                centerweight=centerweight,global_species=global_species,
                               nocenters=nocenters)
        self.fast_avg = fast_avg
        self.is_sparse = is_sparse
        self.disable_pbar = disable_pbar
    def fit(self,X):
        return self
    def get_params(self,deep=True):
        params = dict(is_sparse=self.is_sparse,disable_pbar=False,fast_avg=self.fast_avg)
        params.update(**self.soap_params)
        return params

    def compute_neigbourlist(self,frame):
        frame = ase2qp(frame)
        frame.set_cutoff(self.soap_params['rc'])
        frame.calc_connect()
        return frame

    def init_data(self, frames):
        soap_params = self.soap_params
        chunk_len = 100
        fast_avg = self.fast_avg
        with_gradients = soap_params['grad']
        global_species = soap_params['global_species']
        nocenters = soap_params['nocenters']

        self.atomic_types = [[]]*len(frames)
        for iframe, frame in enumerate(frames):
            numbers = frame.get_atomic_numbers()
            self.atomic_types[iframe] = numbers


        Nfeature = get_Nsoap(global_species,soap_params['nmax'],
                                soap_params['lmax'])

        Ncenter,self.slices,strides = get_frame_slices(frames,nocenters=nocenters, fast_avg=fast_avg)

        if with_gradients is False:
            self.slices_gradients = [None] * len(frames)
            features = FeatureWithGrad(Ncenter=Ncenter, Nfeature=Nfeature, chunk_len=chunk_len,hyperparams=soap_params)
        elif with_gradients is True:
            Nneighbour,strides_gradients,self.slices_gradients = get_frame_neigbourlist(frames,nocenters=nocenters)

            features = FeatureWithGrad(Ncenter=Ncenter, Nfeature=Nfeature,  Nneighbour=Nneighbour, has_gradients=True,chunk_len=chunk_len,hyperparams=soap_params)

        return features

    def insert_to(self, features, iframe, quippy_results):
        quippy_rep = quippy_results['descriptor']
        quippy_grad,grad_species = None,None
        grad_mapping = None
        desc_mapping = np.zeros((quippy_rep.shape[0],1))
        atom_mapping = quippy_results['descriptor_index_0based'].flatten()
        species = self.atomic_types[iframe]

        if 'grad' in quippy_results:
            quippy_grad = quippy_results['grad']
            grad_mapping = quippy_results['grad_index_0based']
            # pos_ids = np.unique(grad_mapping[:,1])
            # map_grad2desc = np.zeros(quippy_grad.shape[0])
            grad_species = np.zeros(quippy_grad.shape[0])
            # grad_species_quippy = np.zeros(quippy_grad.shape[0])
            # grad = np.zeros(quippy_grad.shape)
            # st = 0
            # for pos_id in pos_ids:
            #     grad_atom_ids = np.where(grad_mapping[:,1] == pos_id)[0]
            #     nd = st + len(grad_atom_ids)
            #     map_grad2desc[st:nd] = grad_mapping[grad_atom_ids,0]
            #     grad_species[st:nd] = species[atom_mapping[grad_mapping[grad_atom_ids,0]]]
            #     #grad_species[st:nd] = species[atom_grad_id]
            #     grad[st:nd] = quippy_grad[grad_atom_ids]
            #     st = nd
            for ii,idesc in enumerate(grad_mapping[:,0]):
                grad_species[ii] = species[idesc]

        features.insert(
                self.slices[iframe], quippy_rep, species, desc_mapping,
                self.slices_gradients[iframe], quippy_grad, grad_species,
                grad_mapping
        )
        # features.insert(
        #         self.slices[iframe], quippy_rep, species,
        #         self.slices_gradients[iframe], quippy_grad, grad_species,
        #         map_grad2desc,grad_mapping,grad_species_quippy
        # )
        # features.insert(
        #         self.slices[iframe], quippy_rep, species,
        #         self.slices_gradients[iframe], grad, grad_species,
        #         map_grad2desc,grad_mapping,grad_species_quippy
        # )
        # features.insert(
        #         self.slices[iframe], quippy_rep, species,
        #         self.slices_gradients[iframe], grad, grad_species,
        #         map_grad2desc
        # )

    def transform(self,X):

        frames = map(self.compute_neigbourlist,X)

        soapstr = Template(' '.join(['average=F normalise=T soap cutoff_dexp=$cutoff_dexp cutoff_rate=$cutoff_rate ',
                            'cutoff_scale=$cutoff_scale central_reference_all_species=F',
                            'central_weight=$centerweight covariance_sigma0=0.0 atom_sigma=$awidth',
                            'cutoff=$rc cutoff_transition_width=$cutoff_transition_width n_max=$nmax l_max=$lmax',
                            'n_species=$nspecies species_Z=$species n_Z=$ncentres Z=$centres']))

        Ncenter,slices,strides = get_frame_slices(frames,nocenters=self.soap_params['nocenters'], fast_avg=self.fast_avg)
        Nframe = len(frames)

        if self.is_sparse:
            soaps = lil_matrix((Nenv,Nsoap),dtype=np.float64)
        else:
            soaps = self.init_data(frames)

        for iframe in tqdm_cs(range(Nframe),desc='RawSoap',leave=True,disable=self.disable_pbar):
            result = get_rawsoap(frames[iframe],soapstr,**self.soap_params)

            if self.is_sparse:
                soap = result['descriptor']
                if self.fast_avg:
                    soap = np.mean(soap,axis=0)
                soap[np.abs(soap)<1e-13] = 0
                soaps[slices[iframe]] = lil_matrix(soap)
            else:
                self.insert_to(soaps,iframe,result)

        if self.is_sparse:
            soaps = soaps.tocsr(copy=False)

        self.slices = slices
        self.strides = strides

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