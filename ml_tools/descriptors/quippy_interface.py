from ..base import AtomicDescriptorBase
from quippy import descriptors
from ..base import np,sp
from scipy.sparse import lil_matrix,csr_matrix   
import re
from string import Template
from ..utils import tqdm_cs

def ase2qp(aseatoms):
    from quippy import Atoms as qpAtoms
    positions = aseatoms.get_positions()
    cell = aseatoms.get_cell()
    numbers = aseatoms.get_atomic_numbers()
    pbc = aseatoms.get_pbc()
    return qpAtoms(numbers=numbers,cell=cell,positions=positions,pbc=pbc)

def get_chunks(lenght,chunk_lenght):
    Nchunk = lenght // chunk_lenght
    rest = lenght % chunk_lenght
    strides = [s*chunk_lenght for s in range(Nchunk+1)]
    if rest > 0:
        Nchunk += 1
        strides.append(strides[-1]+rest)
    bounds = [(s,e) for s,e in zip(strides[:-1],strides[1:])]
    return bounds

def get_frame_slices(frames,nocenters=None,fast_avg=False):
    if fast_avg:
        slices = []
        strides = [0]
        for frame in frames:
            strides.append(1)
        strides = np.cumsum(strides)
        for st,nd in zip(strides[:-1],strides[1:]):
            slices.append(slice(st,nd))
    else:
        slices = []
        strides = [0]
        for frame in frames:
            numbers = frame.get_atomic_numbers()
            if nocenters is not None:
                numbers = [z for z in numbers if z not in nocenters]
            strides.append(len(numbers))
        strides = np.cumsum(strides)
        for st,nd in zip(strides[:-1],strides[1:]):
            slices.append(slice(st,nd))
    return slices,strides

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
                 cutoff_dexp, cutoff_scale,cutoff_rate):
    frame = ase2qp(frame)
    frame.set_cutoff(rc)
    frame.calc_connect()
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
    soap = desc.calc(frame, grad=False)['descriptor']
    return soap  

class RawSoapQUIP(AtomicDescriptorBase):
    def __init__(self,global_species=None,nocenters=None,rc=None, nmax=None, 
                 lmax=None, awidth=None,centerweight=None,cutoff_transition_width=None, cutoff_rate=None,
                 cutoff_dexp=None, cutoff_scale=None,fast_avg=False,is_sparse=False,disable_pbar=False):
        if global_species is None and rc is None:
            pass
        self.soap_params = dict(rc=rc, nmax=nmax, lmax=lmax, awidth=awidth,cutoff_rate=cutoff_rate,
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
    
    def transform(self,X):
        
        frames = X
        
        slices,strides = get_frame_slices(frames,nocenters=self.soap_params['nocenters'],
                                          fast_avg=self.fast_avg )

        Nenv = strides[-1]

        soapstr = Template(' '.join(['average=F normalise=T soap cutoff_dexp=$cutoff_dexp cutoff_rate=$cutoff_rate ',
                            'cutoff_scale=$cutoff_scale central_reference_all_species=F',
                            'central_weight=$centerweight covariance_sigma0=0.0 atom_sigma=$awidth',
                            'cutoff=$rc cutoff_transition_width=$cutoff_transition_width n_max=$nmax l_max=$lmax', 
                            'n_species=$nspecies species_Z=$species n_Z=$ncentres Z=$centres']))
        print soapstr        
        
        Nsoap = get_Nsoap(self.soap_params['global_species'],self.soap_params['nmax'],
                          self.soap_params['lmax'])

        Nframe = len(frames)

        if self.is_sparse:
            soaps = lil_matrix((Nenv,Nsoap),dtype=np.float64)
        else:
            soaps = np.empty((Nenv,Nsoap))

        for iframe in tqdm_cs(range(Nframe),desc='RawSoap',leave=True,disable=self.disable_pbar):
            soap = get_rawsoap(frames[iframe],soapstr,**self.soap_params)
                        
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
