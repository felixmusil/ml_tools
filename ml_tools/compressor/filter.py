from ..base import np,sp
from ..base import BaseEstimator,TransformerMixin
import spglib as spg

class SymmetryFilter(BaseEstimator,TransformerMixin):
    def __init__(self,threshold=1e-4,species=[1]):
        self.threshold = threshold
        self.species = species
    
    def get_params(self,deep=True):
        params = dict(threshold=self.threshold)
        return params
    
    def fit(self,X):
        frames = X
        self.filter_dict = {}
        self.filter_mask = []
        self.filter_ids_inv = []
        self.filter_sp_map = {}
        self.strides = [0]
        for ii,frame in enumerate(frames):
            numbers = frame.get_atomic_numbers()
            self.filter_sp_map[ii] = {}
            Nat = 0
            for jj,sp in enumerate(numbers):
                if sp in self.species:
                    self.filter_sp_map[ii][jj] = Nat
                    Nat += 1
                    
            data = spg.get_symmetry_dataset(frame,symprec=self.threshold)
            equivalent_atoms = data['equivalent_atoms']
            
            equivalent_atoms_ids_unique, indices_inv = np.unique(equivalent_atoms,return_inverse=True)
            
            Nat = 0
            self.filter_dict[ii] = []
            for jj,sp in enumerate(numbers):
                if sp in self.species:
                    #self.strides[-1]+
                    self.filter_ids_inv.append(self.strides[-1]+
                                               indices_inv[
                                                   self.filter_sp_map[ii][
                                                       equivalent_atoms[jj]]])
                    if jj in equivalent_atoms_ids_unique:
                        Nat += 1
                        self.filter_dict[ii].append(jj)
                        self.filter_mask.append(True)
                    else:
                        self.filter_mask.append(False)
            self.strides.append(self.strides[-1]+Nat)
        
        self.Nsample = self.strides[-1] 
        self.filter_mask = np.array(self.filter_mask)
        self.filter_ids_inv = np.array(self.filter_ids_inv)
        return self
    
    def transform(self,X,y):
        feature_matrix = X
        return feature_matrix[self.filter_mask,:],y[self.filter_mask]
    
    def fit_transform(self,X,y):
        frames, feature_matrix = X['frames'],X['feature_matrix']
        self.fit(frames)
        return self.transform(feature_matrix,y)
    
    def inverse_transform(self,X=None,y=None):
        X_full,y_full = None,None
        if y is not None:
            y_full = y[self.filter_ids_inv]
        if X is not None:
            X_full = X[self.filter_ids_inv,:]
        return X_full,y_full
