
import json
from .base import np,sp

class BoxList(object):
    '''
    Example of use:
    list_box = BoxList(self.max_cutoff,self.cell,pbc,self.positions)
    neighlist = [[] for it in range(self.Natom)]
    neighpos = [[] for it in range(self.Natom)]
    neighshift = [[] for it in range(self.Natom)]
    for box in list_box.iter_box():
        for icenter in box.icenters:
            for jneigh,box_shift in box.iter_neigh_box():
            
                nnp = self.positions[jneigh] + np.dot(box_shift.reshape((1,3)),self.cell).reshape((1,3))
                #rr = nnp - self.positions[icenter].reshape((1,3))
                #dist = np.linalg.norm(rr,axis=1)
                neighpos[icenter].extend(nnp)
                neighlist[icenter].append(jneigh)
                neighshift[icenter].append(box_shift)
    '''
    def __init__(self,max_cutoff,cell,pbc,centers):
        # Compute reciprocal lattice vectors.
        b1_c, b2_c, b3_c = np.linalg.pinv(cell).T
        
        # Compute distances of cell faces (height between 2 consecutive faces [010] 
        l1 = np.linalg.norm(b1_c)
        l2 = np.linalg.norm(b2_c)
        l3 = np.linalg.norm(b3_c)
        face_dist_c = np.array([1 / l1 if l1 > 0 else 1,
                                1 / l2 if l2 > 0 else 1,
                                1 / l3 if l3 > 0 else 1])
        
        # We use a minimum bin size of 3 A
        self.bin_size = max_cutoff
        # Compute number of bins such that a sphere of radius cutoff fit into eight
        # neighboring bins.
        self.nbins_c = np.maximum((face_dist_c / self.bin_size).astype(int), [1, 1, 1])
        self.nbins = np.prod(self.nbins_c)
        # Compute over how many bins we need to loop in the neighbor list search.
        self.neigh_search = np.ceil(self.bin_size * self.nbins_c / face_dist_c).astype(int)
        self.bin2icenters = [[] for bin_idx in range(self.nbins)]
        scaled_positions_ic = np.linalg.solve(cell.T,centers.T).T
        self.h_sizes = np.linalg.norm(cell,axis=1)
        self.part2bin = {}
        for icenter in range(len(centers)):
            bin_index_ic = np.floor(scaled_positions_ic[icenter]*self.nbins_c).astype(int)
            bin_id = self.cell2lin(bin_index_ic)
            self.bin2icenters[bin_id].append(icenter)
            self.part2bin[icenter] = bin_id
        self.list = []
        for bin_id in range(self.nbins):
            self.list.append(Box(bin_id,self.nbins_c,self.neigh_search,self.bin2icenters[bin_id],pbc,self))
            
    def cell2lin(self,ids):
        return int(ids[0] + self.nbins_c[0] * (ids[1] + self.nbins_c[1] * ids[2]))
    
    def iter_box(self):
        for bin_id in range(self.nbins):
            yield self.list[bin_id]
    def __getitem__(self, bin_id):
        return self.list[bin_id]
    
class Box(object):
    def __init__(self,lin_pos,nbins_c,neigh_search,icenters,pbc,boxlist):
        self.nbins_c = nbins_c
        self.neigh_search = neigh_search
        self.icenters = icenters
        self.pbc = pbc
        self.lin_pos = lin_pos
        self.mult_pos = self.lin2cell(lin_pos)
        self.boxlist = boxlist
        self.search_idx = []
        for ii in range(3):
            p = self.pbc[ii]
            
            if 0 == self.mult_pos[ii] and p is False:
                self.search_idx.append([self.mult_pos[ii]+jj for jj in range(self.neigh_search[ii]+1)])
            elif self.nbins_c[ii]-1 == self.mult_pos[ii] and p is False:
                self.search_idx.append([self.mult_pos[ii]+jj for jj in range(-self.neigh_search[ii],0+1)])
            else:
                self.search_idx.append([self.mult_pos[ii]+jj for jj in range(-self.neigh_search[ii], self.neigh_search[ii]+1)])
        self.neighbour_bin_index,self.neighbour_bin_shift = [],[]
        for ii in self.search_idx[0]:
            for jj in self.search_idx[1]:
                for kk in self.search_idx[2]:
                    box_shift,box_pos = np.divmod([ii,jj,kk],self.nbins_c)
                    neigh_box_idx = self.cell2lin(box_pos)
                    self.neighbour_bin_index.append(neigh_box_idx)
                    self.neighbour_bin_shift.append(box_shift)
        
                    
    def cell2lin(self,ids):
        return int(ids[0] + self.nbins_c[0] * (ids[1] + self.nbins_c[1] * ids[2]))
    def lin2cell(self,lin_ids):
        fac = 1
        cell_pos = np.array([0,0,0])
        for ii in range(3):
            cell_pos[ii] = lin_ids/fac % self.nbins_c[ii]
            fac *= self.nbins_c[ii]
        return cell_pos
    def iter_neigh_box(self):
        from copy import deepcopy
        for ii in self.search_idx[0]:
            for jj in self.search_idx[1]:
                for kk in self.search_idx[2]:
                    box_shift,box_pos = np.divmod([ii,jj,kk],self.nbins_c)
                    neigh_box_idx = self.cell2lin(box_pos)
                    jcenters = deepcopy(self.boxlist[neigh_box_idx].icenters)
                    for jneigh in jcenters:
                        yield jneigh,deepcopy(box_shift)