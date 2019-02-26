import numpy as np

class DenseFeature(object):
    def __init__(self,representations,gradients=None,neighbourlist=None,strides=None):
        self.representations = representations
        self.gradients = gradients
        self.neighbourlists = neighbourlists
        # each row match a different property
        if strides is None:
            self.strides = np.arange(representations.shape[0])
        else:
            self.strides = np.asarray(strides,dtype=np.int32)

