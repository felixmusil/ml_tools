import numpy as np
from ..base import FeatureBase
# from collections.abc import Iterable

class DenseFeature(FeatureBase):
    def __init__(self, Ncenter, Nfeature, strides, Nneighbour=None, strides_gradients=None, with_gradients=False):

        self.shape_rep = (Ncenter,Nfeature)
        self.representations = np.empty(self.shape_rep)

        self.strides = np.asarray(strides,dtype=np.int32)
        self.Nstructures = len(self.strides)-1
        self.chunk_len = None
        # map frame id and center id to pos in representations
        self.representations_mapping = [[]]*self.Nstructures

        self.with_gradients = with_gradients
        if with_gradients is True:
            self.shape_grad = (3,Nneighbour,Nfeature)
            self.shape_grad_out = (3*Nneighbour,Nfeature)
            self.gradients = np.empty(self.shape_grad)
            self.strides_gradients = np.asarray(strides_gradients,dtype=np.int32)
            # map frame id and center id to slices in gradients
            self.gradients_mapping = [[]]*self.Nstructures
            # map frame id and center id to list of neighbour atoms id
            self.neighbourlists = [[]]*self.Nstructures

    def get_data(self,gradients=False):
        if gradients is False:
            return self.representations,self.strides

        # the xyz compotent is flatten so its is like having 3 times the number
        # of neighbours for the stride. slicing with a step of 3
        # to get only x/y/z related gradients
        elif gradients is True:
            return self.gradients.reshape(self.shape_grad_out),self.strides_gradients

    def insert(self, iframe, slice_representations, representations, neighbourlist=None, slice_gradients=None, gradients=None):
        self.representations[slice_representations] = representations
        for i_center in range(slice_representations.start,
                              slice_representations.stop):
            self.representations_mapping[iframe].append(i_center)

        if self.with_gradients is True:
            self.neighbourlists[iframe] = neighbourlist
            st = slice_gradients.start
            for i_center_local,neigh_ids in enumerate(neighbourlist[iframe]):
                nd = st + len(neigh_ids)

                self.gradients[:,slice_gradients] = np.swapaxes(gradients,1,0)
                self.gradients_mapping[iframe].append(slice(st,nd))
                st = nd

    def extract_pseudo_input(self,ids_pseudo_input):
        Ncenter = len(ids_pseudo_input)
        #print Ncenter
        Nfeature = self.shape_rep[1]
        strides = np.arange(Ncenter+1)
        X_pseudo = DenseFeature(Ncenter,Nfeature,strides)
        X_pseudo.representations_mapping = None
        X_pseudo.representations = self.representations[ids_pseudo_input]
        return X_pseudo

    def extract_feature_selection(self,ids_selected_features):
        Ncenter = self.shape_rep[0]
        Nfeature = len(ids_selected_features)
        strides = self.strides

        if self.with_gradients is True:
            Nneighbour = self.shape_grad[1]
        else:
            Nneighbour = None

        X_selected = DenseFeature(Ncenter,Nfeature,strides,Nneighbour, self.with_gradients)
        X_selected.representations = self.representations[:,ids_selected_features]
        X_selected.representations_mapping = self.representations_mapping

        if self.with_gradients is True:
            X_selected.gradients = self.gradients[:,:,ids_selected_features]
            X_selected.gradients_mapping = self.gradients_mapping
            X_selected.neighbourlists = self.neighbourlists
        return X_selected

    def get_nb_sample(self,gradients=False):
        if gradients is False:
            return len(self.strides)-1
        if gradients is True:
            return len(self.strides_gradients)-1

    def get_nb_environmental_elements(self,gradients=False):
        if gradients is False:
            return self.shape_rep[0]
        if gradients is True:
            return self.shape_grad_out[0]

    def set_chunk_len(self, chunk_len):
        if chunk_len is None:
            pass
        elif chunk_len > self.shape_rep[0]:
            chunk_len = self.shape_rep[0]

        self.chunk_len = chunk_len

    def get_chunk_nb(self,gradients=False):
        if self.chunk_len is None:
            Nchunk = 1
        elif gradients is False:
            N = len(self.strides)-1
            Nchunk = N // self.chunk_len
        elif gradients is True:
            N = len(self.strides_gradients)-1
            Nchunk = N // self.chunk_len
        return Nchunk

    def get_iterator(self,gradients=False):
        if self.chunk_len is None:
            Nsample = self.get_nb_sample(gradients)
            if gradients is False:
                yield self.representations, self.strides, slice(0,Nsample)
            if gradients is True:
                yield self.gradients.reshape(self.shape_grad_out), self.strides_gradients, slice(0,Nsample)
        else:
            if gradients is False:
                frame_ids,chuncks,chuncks_global = get_chunck(self.strides,self.chunk_len)
                for fids1,ch1,ch1g in zip(frame_ids,chuncks,chuncks_global):
                    sl1 = slice(ch1g[0][0],ch1g[-1][1])
                    strides_local = [st for st,nd in ch1]+[ch1[-1][1]]
                    yield self.representations[sl1], strides_local, fids1
            if gradients is True:
                frame_ids,chuncks,chuncks_global = get_chunck(self.strides_gradients,self.chunk_len)
                for fids1,ch1,ch1g in zip(frame_ids,chuncks,chuncks_global):
                    sl1 = slice(ch1g[0][0],ch1g[-1][1])
                    strides_local = [st for st,nd in ch1]+[ch1[-1][1]]
                    grad = self.gradients.reshape(self.shape_grad_out)
                    yield grad[sl1], strides_local, fids1


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