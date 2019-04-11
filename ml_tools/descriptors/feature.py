import numpy as np
from ..base import FeatureBase, is_npy
from ..utils import return_deepcopy
# from collections.abc import Iterable


class DenseFeature(FeatureBase):
    def __init__(self, Ncenter=None, Nfeature=None,has_gradients=False, chunk_len=None,tag='rep',
     data=None,structure_strides=None, species=None,mapping=None):
        if data is None:
            self.Ncenter = Ncenter
            self.Nfeature = Nfeature
            if has_gradients is False:
                self.shape = (Ncenter,Nfeature)
                self.shape_out = (Ncenter,Nfeature)
                self.mapping = -1*np.ones((Ncenter,1),dtype=int)
            elif has_gradients is True:
                self.shape = (Ncenter,3,Nfeature)
                self.shape_out = (Ncenter,3,Nfeature)
                self.mapping = -1*np.ones((Ncenter,3),dtype=int)
            self.data = np.empty(self.shape)
            self.structure_strides = [0]
            self.is_slice = False
            self.species = np.zeros((Ncenter,1),dtype=int)
        else:
            self.data = data
            self.shape = data.shape
            self.Ncenter = data.shape[0]
            self.Nfeature = data.shape[1]
            if structure_strides is None or species is None or mapping is None:
                raise RuntimeError()
            self.mapping = mapping
            self.structure_strides = structure_strides
            self.species = species

            self.is_slice = True
            self.shape_out = self.shape
        self.tag = tag
        self.chunk_len = chunk_len

        self.has_gradients = has_gradients

    @return_deepcopy
    def get_params(self):
        params = {}
        params['Ncenter'] = self.Ncenter
        params['Nfeature'] = self.Nfeature
        params['has_gradients'] = self.has_gradients
        params['chunk_len'] = self.chunk_len
        params['tag'] = self.tag
        return params

    def dumps(self):
        from copy import deepcopy
        state = {}
        state['init_params'] = deepcopy(self.get_params())
        state['data'] = dict(
            data=self.data,
            structure_strides=deepcopy(self.structure_strides),
            tag=self.tag,
            species=deepcopy(self.species),
            mapping=deepcopy(self.mapping),
            )
        return state

    def loads(self, data):
        self.data = data['data']
        self.structure_strides = data['structure_strides']
        self.tag = data['tag']
        self.species = data['species']
        self.mapping = data['mapping']

    def get_nb_frame(self):
        if self.has_gradients is True:
            return self.mapping[:,2].max()+1
        elif self.has_gradients is False:
            return self.mapping[:,0].max()+1

    def get_species(self):
        return self.species

    def get_data(self, specie=None):
        if specie is None:
            data = self.data
        else:
            sp_mask = self.species.flatten() == specie
            data = self.data[sp_mask]
        return data

    def get_mapping(self, specie=None):
        if specie is None:
            mapping = self.mapping
        else:
            sp_mask = self.species.flatten() == specie
            mapping = self.mapping[sp_mask]
        return mapping

    def get_ids(self,is_global=True):
        if is_global is True:
            ids = np.concatenate([self.mapping,self.species],axis=1)
        elif is_global is False and self.has_gradients is False:
            Ncenter = self.shape[0]
            ids = np.concatenate([np.arange(Ncenter)[:,None],self.species],axis=1)
        else:
            raise NotImplementedError()

        return np.asarray(ids,dtype=int)

    def get_nb_elements(self,strides,specie=None):
        if specie is None:
            sp_mask = np.ones(self.species.shape, dtype=int)
        else:
            sp_mask = np.array(self.species.flatten() == specie,dtype=int)

        Nel = [0]
        for st,nd in zip(strides[:-1],strides[1:]):
            n_el = np.sum(sp_mask[st:nd])
            Nel.append(n_el)
        return np.array(Nel)

    def get_structure_strides(self,specie=None):
        if specie is None:
            return np.array(self.structure_strides)

        Nel = self.get_nb_elements(self.structure_strides,specie)
        return np.cumsum(Nel)

    def insert(self, data_slice, data, species, mapping):
        n_rows = data_slice.stop - data_slice.start
        st = self.structure_strides[-1]
        structure_stride = st + n_rows


        self.structure_strides.append(structure_stride)

        Nsample = self.get_nb_sample()

        self.species[data_slice] = species[:,None]
        self.mapping[data_slice] = mapping
        self.mapping[data_slice,0] += Nsample
        self.data[data_slice] = data

    def get_nb_sample(self,specie=None):
        mapping = self.get_mapping(specie)
        return len(np.unique(mapping[mapping[:,0] > -1,0]))

    def get_nb_feature(self):
        return self.shape_out[-1]

    def get_nb_environmental_elements(self,specie=None):
        if specie is None:
            return self.shape_out[0]
        else:
            sp_mask = np.array(self.species.flatten() == specie, dtype=int)
            return np.sum(sp_mask)

    def set_chunk_len(self, chunk_len):
        if chunk_len is None:
            pass
        elif chunk_len > self.shape_out[0]:
            chunk_len = self.shape_out[0]

        self.chunk_len = chunk_len

    def get_chunk_nb(self):
        if self.chunk_len is None:
            Nchunk = 1
        else:
            N = len(self.structure_strides)-1
            Nchunk = N // self.chunk_len
            if N % self.chunk_len != 0:
                Nchunk += 1
        return Nchunk

    def get_structure_slices(self):
        chunk_len = self.chunk_len
        Nstructures = len(self.structure_strides)-1
        Nchunk = Nstructures // chunk_len
        structure_slices = [slice(it*chunk_len,(it+1)*chunk_len) for it in range(Nchunk)]
        if Nstructures % chunk_len != 0:
            structure_slices.append(slice((Nchunk)*chunk_len,(Nstructures)))
        return structure_slices

    def get_iterator(self):
        if self.chunk_len is None:
            yield self
        else:
            structure_slices = self.get_structure_slices()
            for structure_slice in structure_slices:
                yield self[structure_slice]

    def get_data_slice(self, structure_slice):
        st,nd = structure_slice.start, structure_slice.stop+1
        global_strides = self.structure_strides[st:nd]
        if len(global_strides) < 2:
            global_strides = [0, 1]
        data_slice = slice(global_strides[0],global_strides[-1])
        return data_slice

    def __getitem__(self, structure_slice):
        if isinstance(structure_slice, int):
            structure_slice = slice(structure_slice,structure_slice+1)

        data_slice = self.get_data_slice(structure_slice)
        st = structure_slice.start
        nd = structure_slice.stop + 1
        structure_strides = self.structure_strides[st:nd]

        data = self.data[data_slice]
        species = self.species[data_slice]
        mapping = self.mapping[data_slice]
        obj_slice = DenseFeature(data=data,
                                mapping=mapping,
                                has_gradients=self.has_gradients,
                                chunk_len=self.chunk_len,
                                structure_strides=structure_strides,
                                species=species)
        return obj_slice


def get_chunck(strides,chunk_len):
    N = len(strides)-1
    Nchunk = N // chunk_len

    slices = [(st,nd) for st,nd in zip(strides[:-1],strides[1:])]

    if Nchunk == 1 and N % chunk_len == 0:
        return [slice(0,N)],[slices],[slices]

    frame_ids = [slice(it*chunk_len,(it+1)*chunk_len) for it in range(Nchunk)]
    if N % chunk_len != 0:
        frame_ids.append(slice((Nchunk)*chunk_len,(N)))

    chuncks_global = [
        [slices[ii] for ii in range(it*chunk_len,(it+1)*chunk_len)]
              for it in range(Nchunk)]
    if N % chunk_len != 0:
        chuncks_global.append([slices[ii] for ii in range((Nchunk)*chunk_len,(N))])

    chuncks = []
    for it in range(Nchunk):
        st_ref = slices[it*chunk_len][0]
        aa = []
        for ii in range(it*chunk_len,(it+1)*chunk_len):
            st,nd = slices[ii][0]-st_ref,slices[ii][1]-st_ref
            aa.append((st,nd))
        chuncks.append(aa)

    if N % chunk_len != 0:
        aa = []
        st_ref = slices[Nchunk*chunk_len][0]
        for ii in range((Nchunk)*chunk_len,N):
            st,nd = slices[ii][0]-st_ref,slices[ii][1]-st_ref
            aa.append((st,nd))
        chuncks.append(aa)
    return frame_ids,chuncks,chuncks_global


class Representation(FeatureBase):
    def __init__(self, representations=None, gradients=None, Ncenter=None, Nfeature=None, Nneighbour=None, species=None, has_gradients=False,chunk_len=None,hyperparams=None):
        self.chunk_len = chunk_len
        self.hyperparams = hyperparams
        self.Ncenter = Ncenter
        self.Nfeature = Nfeature
        self.Nneighbour = Nneighbour

        if representations is None:
            self.representations = DenseFeature(Ncenter=Ncenter, Nfeature=Nfeature, has_gradients=False, chunk_len=chunk_len,tag='rep')
        else:
            self.representations = representations

        self.has_gradients = has_gradients

        if has_gradients is True:
            if gradients is None:
                self.gradients = DenseFeature(Ncenter=Nneighbour, Nfeature=Nfeature, has_gradients=True, chunk_len=chunk_len,tag='grad')
            else:
                self.gradients = gradients
        else:
            self.gradients = None

    def __getitem__(self, structure_slice):
        rep = self.representations[structure_slice]
        Ncenter = rep.shape[0]
        if self.has_gradients is True:
            grad = self.gradients[structure_slice]
            Nneighbour = grad.shape[0]
        else:
            Nneighbour = self.Nneighbour
            grad = None

        obj_slice = Representation(chunk_len=self.chunk_len,
                                    hyperparams=self.hyperparams,
                                    Ncenter=Ncenter,
                                    Nfeature=self.Nfeature,
                                    Nneighbour=Nneighbour,
                                    has_gradients=self.has_gradients,
                                    representations=rep,
                                    gradients=grad)
        return obj_slice

    def get_species(self,gradients=False):
        if gradients is False:
            return self.representations.get_species()
        elif gradients is True:
            return self.gradients.get_species()

    def get_data(self,gradients=False,specie=None):
        if gradients is False:
            return self.representations.get_data(specie)
        elif gradients is True:
            return self.gradients.get_data(specie)
    def get_mapping(self,gradients=False,specie=None ):
        if gradients is False:
            return self.representations.get_mapping(specie)
        elif gradients is True:
            return self.gradients.get_mapping(specie)

    def get_ids(self,gradients=False,is_global=True):
        if gradients is False:
            return self.representations.get_ids(is_global)
        elif gradients is True:
            return self.gradients.get_ids()


    def insert(self, slice_representations, representations, species_representations, representations_mapping, slice_gradients=None, gradients=None, species_gradients=None, gradient_mapping=None):
        self.representations.insert(slice_representations,representations,species_representations,representations_mapping)
        if self.has_gradients is True:
            self.gradients.insert(slice_gradients,gradients,species_gradients,gradient_mapping)


    def extract_pseudo_input(self, ids_pseudo_input):
        if isinstance(ids_pseudo_input, dict) is True:
            Nfeature = self.representations.get_nb_feature()
            Ncenter = 0
            for sp in ids_pseudo_input:
                Ncenter += len(ids_pseudo_input[sp])

            X_pseudo = Representation(Ncenter=Ncenter,Nfeature=Nfeature)
            st = 0
            for sp in ids_pseudo_input:
                rep = self.representations.get_data(sp)
                for idx in ids_pseudo_input[sp]:
                    nd = st + 1
                    X_pseudo.insert(slice(st,nd),rep[idx],np.array([sp]),0)
                    st = nd
        elif is_npy(ids_pseudo_input) is True:
            Nfeature = self.representations.get_nb_feature()
            Ncenter = ids_pseudo_input.shape[0]

            X_pseudo = Representation(Ncenter=Ncenter,Nfeature=Nfeature)
            rep = self.representations.get_data()
            species = self.representations.get_species()
            st = 0
            for idx in range(Ncenter):
                nd = st + 1
                X_pseudo.insert(slice(st,nd),rep[idx],np.array([species[idx]]),0)
                st = nd
        return X_pseudo

    def get_nb_frame(self):
        return self.representations.get_nb_frame()

    def get_local_view(self):
        # works only for the representation (not the gradients)
        Nframes = self.get_nb_frame()
        obj = self[:Nframes]
        Ncenters = obj.get_nb_environmental_elements()
        # obj.representations.mapping = np.arange(Ncenters)[:,None]
        obj.representations.structure_strides = np.arange(Ncenters+1)
        return obj

    def extract_feature_selection(self,ids_selected_features):
        Nfeature = len(ids_selected_features)
        rep = self.representations.get_data()
        new_rep = rep[:,ids_selected_features]
        Ncenter = rep.shape[0]
        ids = self.get_ids()
        X_selected = Representation(Ncenter=Ncenter,Nfeature=Nfeature)
        st = 0
        for icenter,(iframe,sp) in enumerate(ids):
            nd = st + 1
            X_selected.insert(slice(st,nd),new_rep[icenter],np.array([sp]),0)
            st = nd

        return X_selected

    def get_transpose(self):
        rep = self.representations.get_data()
        Ncenter,Nfeature = rep.shape
        ids = self.get_ids()
        X = Representation(Ncenter=Nfeature,Nfeature=Ncenter)
        st = 0
        for ifeature in range(Nfeature):
            nd = st + 1
            X.insert(slice(st,nd),rep[:,ifeature],np.array([0]),0)
            st = nd
        return X

    T = property(get_transpose)

    def get_nb_sample(self,gradients=False,specie=None,is_global=True):
        if is_global is True:
            if gradients is False:
                return self.representations.get_nb_sample(specie)
            elif gradients is True:
                return self.gradients.get_nb_sample(specie)
        elif is_global is False:
            return self.get_nb_environmental_elements(specie,gradients)

    def get_nb_environmental_elements(self,specie=None,gradients=False):
        if gradients is False:
            return self.representations.get_nb_environmental_elements(specie)
        if gradients is True:
            return self.gradients.get_nb_environmental_elements(specie)

    def set_chunk_len(self, chunk_len):
        if chunk_len is None:
            pass
        elif chunk_len > self.get_nb_sample():
            chunk_len = self.get_nb_sample()

        self.representations.set_chunk_len(chunk_len)
        if self.has_gradients is True:
            self.gradients.set_chunk_len(chunk_len)
        self.chunk_len = chunk_len

    def get_chunk_nb(self,gradients=False):
        if self.chunk_len is None:
            Nchunk = 1
        elif gradients is False:
            Nchunk = self.representations.get_chunk_nb()
        elif gradients is True:
            Nchunk = self.gradients.get_chunk_nb()
        return Nchunk

    def get_iterator(self,gradients=False):
        if self.has_gradients is False and gradients is False:
            for rep in self.representations.get_iterator():
                yield Representation(representations=rep,
                                    has_gradients=False,
                                    chunk_len=self.chunk_len,
                                    hyperparams=self.hyperparams)
        elif self.has_gradients is True and gradients is False:
            for rep in self.representations.get_iterator():
                grad = self.gradients[rep.structure_slice]

                yield Representation(representations=rep,
                                    gradients=grad,
                                    has_gradients=True,
                                    chunk_len=self.chunk_len,
                                    hyperparams=self.hyperparams)

        elif self.has_gradients is True and gradients is True:
            for grad in self.gradients.get_iterator():
                rep = self.representations[grad.structure_slice]

                feat = Representation(representations=rep,
                                    gradients=grad,
                                    has_gradients=True,
                                    chunk_len=self.chunk_len,
                                    hyperparams=self.hyperparams)
                yield feat
        else:
            raise RuntimeError('gardients is {} but has_gradients is {}'.format(gradients,self.has_gradients))

    @return_deepcopy
    def get_params(self):
        params = {}
        params['Ncenter'] = self.Ncenter
        params['Nfeature'] = self.Nfeature
        params['Nneighbour'] = self.Nneighbour
        params['has_gradients']= self.has_gradients
        params['chunk_len'] = self.chunk_len
        params['hyperparams'] = self.hyperparams
        return params

    def dumps(self):
        from copy import deepcopy
        state = {}
        state['init_params'] = deepcopy(self.get_params())
        state['data'] = dict(representations=self.representations,
                            gradients=self.gradients)

        return state

    def loads(self, data):
        self.representations = data['representations']
        self.gradients = data['gradients']
