import numpy as np

class DenseFeature(object):
    def __init__(self,Ncenter,Nfeature,strides, Nneighbour=None, with_gradients=False):

        self.representations = np.empty((Ncenter,Nfeature))
        self.strides = np.asarray(strides,dtype=np.int32)
        self.Nstructures = len(self.strides)-1
        # map frame id and center id to pos in representations
        self.representations_mapping = [[]]*self.Nstructures
        self.shape_rep = (Ncenter,Nfeature)
        self.with_gradients = with_gradients
        if with_gradients is True:
            self.gradients = np.empty((Nneighbour,3,Nfeature))
            # map frame id and center id to slices in gradients
            self.gradients_mapping = [[]]*self.Nstructures
            # map frame id and center id to list of neighbour atoms id
            self.neighbourlists = [[]]*self.Nstructures

            self.shape_grad = (Nneighbour,3,Nfeature)

    def insert(iframe, slice_representations, representations, neighbourlist=None, slice_gradients=None, gradients=None):
        self.representations[slice_representations] = rep
        self.neighbourlists[iframe] = neighbourlist
        for i_center in range(slice_representations.start,
                              slice_representations.stop):
            self.representations_mapping[iframe].append(i_center)

        if self.with_gradients is True:
            st = slice_gradients.start
            for i_center_local,neigh_ids in enumerate(neighbourlist):
                nd = st + len(neigh_ids)
                ii_grads = np.where(neighbourlist[:,0] == i_center_local)[0]
                self.gradients[slice_gradients] = gradients
                self.gradients_mapping[iframe].append(slice(st,nd))
                st = nd

    def extract_pseudo_input(ids_pseudo_input):
        Ncenter = len(ids_pseudo_input)
        Nfeature = self.shape_rep[1]
        strides = np.arange(Ncenter)
        X_pseudo = DenseFeature(Ncenter,Nfeature,strides)
        X_pseudo.representations_mapping = None
        X_pseudo.representations = self.representations[ids_pseudo_input]
        return X_pseudo