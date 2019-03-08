import numpy as np
from ..base import FeatureBase
# from collections.abc import Iterable


class DenseFeature(FeatureBase):
    def __init__(self,data=None, Ncenter=None, Nfeature=None, strides=None, has_gradients=False, chunk_len=None , structure_strides=None, structure_slice=None,reduced_data_slice=None):
        if data is None:
            if has_gradients is False:
                self.shape = (Ncenter,Nfeature)
                self.shape_out = (Ncenter,Nfeature)
            elif has_gradients is True:
                self.shape = (3,Ncenter,Nfeature)
                self.shape_out = (3*Ncenter,Nfeature)

            self.data = np.empty(self.shape)
            self.structure_strides = [0]
            self.structure_slice = slice(0,0)
            self.data_slice = slice(0,0)
            self.reduced_data_slice = slice(0,0)
            self.is_slice = False
        else:
            self.data = data
            self.shape = data.shape
            if structure_strides is None or structure_slice is None or reduced_data_slice is None:
                raise RuntimeError()
            self.structure_strides = structure_strides
            self.structure_slice = structure_slice
            self.reduced_data_slice = reduced_data_slice
            self.is_slice = True
            if has_gradients is False:
                self.shape_out = self.shape
            elif has_gradients is True:
                self.shape_out = (3*self.shape[1],self.shape[2])

        self.strides = np.asarray(strides,dtype=np.int32)
        self.Nitems = len(self.strides)-1

        self.chunk_len = chunk_len

        self.has_gradients = has_gradients

    def get_data(self,gradients=None):
        return self.data.reshape(self.shape_out)

    def get_strides(self):
        return self.strides

    def insert(self, data_slice, data):
        n_rows = data_slice.stop - data_slice.start
        st = self.structure_strides[-1]
        structure_stride = st + n_rows
        st = np.where(st == self.strides)[0][0]
        nd = np.where(structure_stride == self.strides)[0][0]
        Nsample = nd - st
        if self.has_gradients is True:
            Nsample = Nsample*3

        self.structure_strides.append(structure_stride)
        self.structure_slice = slice(0,self.structure_slice.stop+1)
        self.reduced_data_slice = slice(0,self.reduced_data_slice.stop+Nsample)

        if self.has_gradients is False:
            self.data[data_slice] = data
        elif self.has_gradients is True:
            self.data[:,data_slice] = np.swapaxes(data,1,0)

    def get_nb_sample(self):
        return self.Nitems

    def get_nb_feature(self):
        return self.shape_out[1]

    def get_nb_environmental_elements(self):
        return self.shape_out[0]

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

    def get_reduced_data_slice(self):
        return self.reduced_data_slice

    def get_iterator(self):
        if self.chunk_len is None:
            yield self
        else:
            structure_slices = self.get_structure_slices()
            for structure_slice in structure_slices:
                yield self[structure_slice]

    def __getitem__(self, structure_slice):
        st,nd = structure_slice.start, structure_slice.stop+1
        global_strides = self.structure_strides[st:nd]
        data_slice = slice(global_strides[0],global_strides[-1])

        structure_strides = self.structure_strides[structure_slice]

        st = np.where(self.strides == data_slice.start)[0][0]
        nd = np.where(self.strides == data_slice.stop)[0][0]
        # print np.where(self.strides == data_slice.stop)
        # print self.strides


        # print structure_slice,reduced_data_slice
        # print global_strides
        # print data.shape
        #print 'sd',strides_local

        strides_local = self.strides[st:nd+1] - self.strides[st]
        reduced_data_slice = slice(st,nd)
        if self.has_gradients is False:
            data = self.data[data_slice]
            # print 'sd',data_slice

        elif self.has_gradients is True:
            data = self.data[:,data_slice]
            strides_local_ = []
            for st,nd in zip(strides_local[:-1],strides_local[1:]):
                strides_local_.append(nd-st)
            strides_local = np.cumsum([0]+strides_local_*3)


        obj_slice = DenseFeature(data=data,
                                strides=strides_local,
                                has_gradients=self.has_gradients,
                                chunk_len=self.chunk_len,
                                structure_strides=structure_strides,
                                structure_slice=structure_slice,
                                reduced_data_slice=reduced_data_slice)
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


class FeatureWithGrad(FeatureBase):
    def __init__(self, representations=None, gradients=None, Ncenter=None, Nfeature=None, strides=None, Nneighbour=None, strides_gradients=None, has_gradients=False,chunk_len=None):
        self.chunk_len = chunk_len

        if representations is None:
            self.representations = DenseFeature(Ncenter=Ncenter, Nfeature=Nfeature, strides=strides, has_gradients=False, chunk_len=chunk_len)
        else:
            self.representations = representations

        self.has_gradients = has_gradients
        if has_gradients is True:
            if gradients is None:
                self.gradients = DenseFeature(Ncenter=Nneighbour, Nfeature=Nfeature, strides=strides_gradients, has_gradients=True, chunk_len=chunk_len)
            else:
                self.gradients = gradients

    def get_data(self,gradients=False):
        if gradients is False:
            return self.representations.get_data()
        elif gradients is True:
            return self.gradients.get_data()

    def get_strides(self,gradients=False):
        if gradients is False:
            return self.representations.get_strides()
        elif gradients is True:
            return self.gradients.get_strides()

    def insert(self, slice_representations, representations, slice_gradients=None, gradients=None):
        self.representations.insert(slice_representations,representations)
        if self.has_gradients is True:
            self.gradients.insert(slice_gradients,gradients)

    def extract_pseudo_input(self, ids_pseudo_input):
        Ncenter = len(ids_pseudo_input)
        Nfeature = self.representations.get_nb_feature()
        strides = np.arange(Ncenter+1)
        rep = self.representations.get_data()
        X_pseudo = FeatureWithGrad(Ncenter=Ncenter,Nfeature=Nfeature,strides=strides)
        st = 0
        for idx in ids_pseudo_input:
            nd = st + 1
            X_pseudo.insert(slice(st,nd),rep[idx])
        return X_pseudo

    def extract_feature_selection(self,ids_selected_features):
        raise NotImplementedError()
        Ncenter = self.representations.get_nb_environmental_elements()
        Nfeature = len(ids_selected_features)
        rep = self.representations.get_data()
        strides = self.representations.get_strides()

        X_selected = FeatureWithGrad(Ncenter=Ncenter,Nfeature=Nfeature,strides=strides)
        X_selected.insert(slice(0,Ncenter),rep[:,ids_selected_features])

        return X_selected

    def get_nb_sample(self,gradients=False):
        if gradients is False:
            return self.representations.get_nb_sample()
        if gradients is True:
            return self.gradients.get_nb_sample()

    def get_nb_environmental_elements(self,gradients=False):
        if gradients is False:
            return self.representations.get_nb_environmental_elements()
        if gradients is True:
            return self.gradients.get_nb_environmental_elements()

    def get_reduced_data_slice(self,gradients=False):
        if gradients is False:
            return self.representations.get_reduced_data_slice()
        if gradients is True:
            return self.gradients.get_reduced_data_slice()

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
                yield FeatureWithGrad(representations=rep,
                                    has_gradients=False,
                                    chunk_len=self.chunk_len)
        elif self.has_gradients is True and gradients is False:
            for rep in self.representations.get_iterator():
                grad = self.gradients[rep.structure_slice]
                yield FeatureWithGrad(representations=rep,
                                    gradients=grad,
                                    has_gradients=True,
                                    chunk_len=self.chunk_len)
        elif self.has_gradients is True and gradients is True:
            for grad in self.gradients.get_iterator():
                rep = self.representations[grad.structure_slice]
                feat = FeatureWithGrad(representations=rep,
                                    gradients=grad,
                                    has_gradients=True,
                                    chunk_len=self.chunk_len)
                yield feat
        else:
            raise RuntimeError('gardients is {} but has_gradients is {}'.format(gradients,self.has_gradients))

