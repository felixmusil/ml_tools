from ..base import AtomicDescriptorBase,FeatureBase
from quippy import descriptors
from ..base import np,sp
from scipy.sparse import lil_matrix,csr_matrix
import re
from string import Template
from ..utils import tqdm_cs
from .utils import get_chunks,get_frame_slices
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

def get_frame_neigbourlist(frames,nocenters):
    Nneighbour = 0
    strides_neighbour = []
    strides_gradients = [0]
    for frame in frames:
        # include centers too wit +1
        numbers = frame.get_atomic_numbers()
        n_neighb = frame.get_array('n_neighb')+1
        mask = np.zeros(numbers.shape,dtype=bool)

        for sp in nocenters:
            mask = np.logical_or(mask, numbers == sp)
        mask = np.logical_not(mask)

        n_neighb = n_neighb[mask]
        Nneighbour += np.sum(n_neighb)
        strides_neighbour += list(n_neighb)
        strides_gradients += [np.sum(n_neighb)]

    strides_gradients = np.cumsum(strides_gradients)
    slices_gradients = []
    for st,nd in zip(strides_gradients[:-1],strides_gradients[1:]):
        slices_gradients.append(slice(st,nd))

    strides_gradients = [0]+strides_neighbour*3

    strides_gradients = np.cumsum(strides_gradients)

    return Nneighbour,strides_gradients,slices_gradients

class QuippyFeature(FeatureBase):
    def __init__(self, frames, soap_params, fast_avg=False, chunk_len=100):
        self.soap_params = soap_params
        self.with_gradients = soap_params['grad']
        global_species = soap_params['global_species']

        self.species_mapping,self.atomic_types = {},{}
        self.slices = {}
        self.features, self.slices_gradients = {},{}

        species = []
        for iframe, frame in enumerate(frames):
            numbers = frame.get_atomic_numbers()
            self.atomic_types[iframe] = numbers
            species.extend(numbers)
        species = np.array(species)

        Nfeature = get_Nsoap(global_species,soap_params['nmax'],
                                soap_params['lmax'])

        for sp in global_species:
            ids = np.where(species == sp)[0]
            self.species_mapping[sp] = np.concatenate([ids,np.arange(len(ids))])
            nocenters = np.setdiff1d(global_species,[sp])

            self.slices[sp],strides = get_frame_slices(frames,nocenters=nocenters, fast_avg=fast_avg)

            Ncenter = strides[-1]

            if self.with_gradients is False:
                self.features[sp] = FeatureWithGrad(Ncenter=Ncenter, Nfeature=Nfeature, strides=strides, chunk_len=chunk_len)
            elif self.with_gradients is True:
                Nneighbour,strides_gradients,self.slices_gradients[sp] = get_frame_neigbourlist(frames,nocenters=nocenters)

                self.features[sp] = FeatureWithGrad(Ncenter=Ncenter, Nfeature=Nfeature, strides=strides, Nneighbour=Nneighbour, strides_gradients=strides_gradients, has_gradients=True,chunk_len=chunk_len)

        self.chunk_len = chunk_len
        self.gradient_mapping = []
        self.Nstructures = len(frames)
        self.current_specie = None
        self.strides_gradient_alternative = {1:[],6:[],8:[]}

    def set_current_specie(self,specie):
        if specie in self.features:
            self.current_specie = specie

    def get_data(self,specie=None,gradients=False):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].get_data(gradients)

    def get_reduced_data_slice(self,specie=None,gradients=False):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].get_reduced_data_slice(gradients)

    def get_iterator(self,specie=None,gradients=False):
        if specie is None:
            specie = self.current_specie

        # print '##############  ',specie,gradients
        # global_rep_slice = self.features[specie].get_reduced_data_slice(gradients)
        # strides = self.features[specie].get_strides(gradients)
        # print self.features[specie].get_data(gradients).shape
        # print strides.shape,strides.min(),strides.max()
        # print global_rep_slice

        for feat in self.features[specie].get_iterator(gradients):
            # global_rep_slice = feat.get_reduced_data_slice(gradients)
            # strides = feat.get_strides(gradients)
            # print '##############12321  '
            # print feat.get_data(gradients).shape
            # print strides.shape,strides.min(),strides.max()
            # print global_rep_slice
            yield feat


    def insert(self, iframe, quippy_results):
        quippy_rep = quippy_results['descriptor']
        quippy_grad = quippy_results['grad']
        grad_mapping = quippy_results['grad_index_0based']
        atom_mapping = quippy_results['descriptor_index_0based'].flatten()
        numbers = self.atomic_types[iframe]

        grad = {}
        for sp in np.unique(numbers):
            grad[sp] = None


        if self.with_gradients is True:
            desc_ids = np.unique(grad_mapping[:,0])
            tmp = {}
            st = 0
            for desc_id in zip(desc_ids):
                atom_grad_id = atom_mapping[desc_id]
                grad_atom_ids = np.where(grad_mapping[:,1] == atom_grad_id)[0]

                nd = st + len(grad_atom_ids)

                self.gradient_mapping.append([])
                tmp[atom_grad_id] = quippy_grad[grad_atom_ids]
                st += nd

            for sp in np.unique(numbers):
                ids = np.sort(np.where(numbers == sp)[0])
                for idx in ids:
                    aa = tmp[idx]
                    self.strides_gradient_alternative[sp].append(aa.shape[0])
                grad[sp] = np.concatenate([tmp[idx] for idx in ids], axis=0)

        for sp in np.unique(numbers):
            ids = np.where(numbers == sp)[0]

            self.features[sp].insert(
                    self.slices[sp][iframe], quippy_rep[ids],
                    self.slices_gradients[sp][iframe], grad[sp]
            )

    def extract_pseudo_input(self, ids_pseudo_input, specie=None):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].extract_pseudo_input(ids_pseudo_input)

    def extract_feature_selection(self,ids_selected_features):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].extract_feature_selection(ids_selected_features)

    def get_nb_sample(self,specie=None,gradients=False):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].get_nb_sample(gradients)

    def get_nb_sample(self,specie=None,gradients=False):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].get_nb_sample(gradients)

    def get_nb_environmental_elements(self,specie=None,gradients=False):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].get_nb_environmental_elements(gradients)

    def get_strides(self,specie=None,gradients=False):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].get_strides(gradients)

    def get_chunk_nb(self,specie=None,gradients=False):
        if specie is None:
            specie = self.current_specie
        return self.features[specie].get_chunk_nb(gradients)

    def set_chunk_len(self,chunk_len):
        for sp in self.features:
            self.features[sp].set_chunk_len(chunk_len)
            self.chunk_len = self.features[sp].chunk_len


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

    def transform(self,X):

        frames = map(self.compute_neigbourlist,X)

        soapstr = Template(' '.join(['average=F normalise=T soap cutoff_dexp=$cutoff_dexp cutoff_rate=$cutoff_rate ',
                            'cutoff_scale=$cutoff_scale central_reference_all_species=F',
                            'central_weight=$centerweight covariance_sigma0=0.0 atom_sigma=$awidth',
                            'cutoff=$rc cutoff_transition_width=$cutoff_transition_width n_max=$nmax l_max=$lmax',
                            'n_species=$nspecies species_Z=$species n_Z=$ncentres Z=$centres']))

        slices,strides = get_frame_slices(frames,nocenters=self.soap_params['nocenters'], fast_avg=self.fast_avg)
        Nframe = len(frames)

        if self.is_sparse:
            soaps = lil_matrix((Nenv,Nsoap),dtype=np.float64)
        else:
            soaps = QuippyFeature(frames, self.soap_params, fast_avg=self.fast_avg)


        for iframe in tqdm_cs(range(Nframe),desc='RawSoap',leave=True,disable=self.disable_pbar):
            result = get_rawsoap(frames[iframe],soapstr,**self.soap_params)

            if self.is_sparse:
                soap = result['descriptor']
                if self.fast_avg:
                    soap = np.mean(soap,axis=0)
                soap[np.abs(soap)<1e-13] = 0
                soaps[slices[iframe]] = lil_matrix(soap)
            else:
                soaps.insert(iframe,result)

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