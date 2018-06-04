#!/usr/bin/python

import sys
import time
import ase
from ase.io import read
import numpy as np
import scipy
from scipy import special
from sympy.physics.wigner import wigner_3j
import argparse
import utils
import random

parser = argparse.ArgumentParser(description="Do PS")
parser.add_argument("-n", "--nmax", type=int, default=8, help="Number of radial functions")
parser.add_argument("-l", "--lmax", type=int, default=6, help="Number of angular functions")
parser.add_argument("-rc", "--rcut", type=float, default=4.0, help="Environment cutoff")
parser.add_argument("-rk", "--rkill", type=float, default=1e-10, help="Killing cutoff")
parser.add_argument("-sg", "--sigma", type=float, default=0.3, help="Gaussian width")
parser.add_argument("-c", "--centres", type=str, default = '', nargs='+', help="List of centres")
parser.add_argument("-cw", "--cweight", type=float, default=1.0, help="Central atom weight")
parser.add_argument("-lm", "--lambdaval", type=int, default=0, help="Spherical tensor order")
parser.add_argument("-p", "--periodic", type=bool, default=False, help="Is the system periodic?")
parser.add_argument("-z", "--zeta", type=int, default=1, help="Kernel exponentiation")
parser.add_argument("-nc", "--ncut", type=int, default=-1, help="Dimensionality cutoff")
parser.add_argument("-sf", "--sparsefile", type=str, default='', help="File with sparsification parameters")
parser.add_argument("-f", "--fname", type=str, required=True, help="Filename")
parser.add_argument("-r", "--randomnum", type=int, default=-1, help="Take this many configurations at random and sparsify on them")
parser.add_argument("-fp", "--fpsnum", type=int, default=-1, help="Take the first set of configurations and sparsify on them")
parser.add_argument("-o", "--outfile", type=str, default='', help="Output file for power spectrum")
args = parser.parse_args()

# SOAP PARAMETERS
nmax = args.nmax              # number of radial functions
lmax = args.lmax              # number of angular functions
rc = args.rcut              # environment cutoff
rk = args.rkill              # killing cutoff
sg = args.sigma              # Gaussian width
cen = ["C","Cl","H","N","O","S"]           # atoms to center on
if args.centres != '':
    cen = args.centres
cw = args.cweight              # central atom weight
lam = args.lambdaval               # spherical tensor order
periodic = args.periodic      # True for periodic systems
zeta = args.zeta              # Kernel exponentiation
ncut = args.ncut            # dimensionality cutoff
sparsefile = args.sparsefile
fname = args.fname
randomnum = args.randomnum
fpsnum = args.fpsnum

sparse_options = [sparsefile]
if sparsefile != '':
    # Here we will read in a file containing sparsification details.
    sparse_fps = np.load("SPARSE_fps"+sparsefile)
    sparse_options.append(sparse_fps)
    sparse_Amatr = np.load("SPARSE_Amat"+sparsefile)
    sparse_options.append(sparse_Amatr)

# BUILT IN PARAMETERS
nnmax = 25            # Max number of neighbours
nsmax = 26            # max number of species (up to iron)
atom_valence = {"H": 1,"He": 2,"Li": 3,"Be": 4,"B": 5,"C": 6,"N": 7,"O": 8,"F": 9,"Ne": 10,"Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15,"S": 16,"Cl": 17,"Ar": 18,"K": 19,"Ca": 20,"Sc": 21,"Ti": 22,"V": 23,"Cr": 24,"Mn": 25,"Fe": 26,"Co": 27,"Ni": 28,"Cu": 29,"Zn": 30} 
atom_symbols = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn'}

# Order the list of centres based on their valence numbers
cen = [atom_symbols[num] for num in list(set([atom_valence[centre] for centre in cen]))]

xyzfile = read(fname,':')

if randomnum > 0:
    print "Shuffling coordinates and taking %i of them."%randomnum
    if sparsefile != '':
        print "Random option is not compatible with supplying a sparsification filename!"
        sys.exit(0)
    if ncut == -1:
        print "Option ncut must be specified for use with random option!"
        sys.exit(0)
    random.shuffle(xyzfile)
    xyzfile = xyzfile[:randomnum]

if fpsnum > 0:
    print "Taking the first %i of the coordinates."%fpsnum
    if sparsefile != '':
        print "This option is not compatible with supplying a sparsification filename!"
        sys.exit(0)
    if ncut == -1:
        print "Option ncut must be specified for use with this option!"
        sys.exit(0)
    xyzfile = xyzfile[:fpsnum]

npoints = len(xyzfile)
all_names = [xyzfile[i].get_chemical_symbols() for i in xrange(npoints)]
coords = [xyzfile[i].get_positions() for i in xrange(npoints)]
if periodic == True:
    cell = [xyzfile[i].get_cell() for i in xrange(npoints)]
else:
    cell = [0.0 for i in xrange(npoints)]


# Process input names
# Here we go by the names in the list of centres, because this one may contain more than we need, but will guarantee we have at least all that we need.
full_names_list = [name for name in cen]
unique_names = list(set(full_names_list))
nspecies = len(unique_names)

# List all species according to their valence
all_species = []
for k in unique_names:
    all_species.append(atom_valence[k])

# List number of atoms for each configuration
natmax = len(max(all_names, key=len))
nat = np.zeros(npoints,dtype=int)
for i in xrange(npoints):
    nat[i] = len(all_names[i])

# List indices for atom of the same species for each configuration
atom_indexes = [[[] for j in xrange(nsmax)] for i in xrange(npoints)]
for i in xrange(npoints):
    for ii in xrange(nat[i]):
        for j in all_species:
            if all_names[i][ii] == atom_symbols[j]:
                atom_indexes[i][j].append(ii)

if len(cen) == 0 :
    centers = all_species
else:
    centers = [atom_valence[i] for i in cen]

# Setup orthogonal matrix.
[orthomatrix,sigma] = utils.setup_orthomatrix(nmax,rc,rk)

#print "Expansion coefficients computed"

llmax = 0
lvalues = {}
if lam > 0: # needed for tensorial stuff
    llmax=0
    lvalues = {}
    for l1 in xrange(lmax+1):
        for l2 in xrange(lmax+1):
            if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
               lvalues[llmax] = [l1,l2] 
               llmax+=1

start = time.time()

# COMPUTE POWER SPECTRUM OF ORDER LAMBDA
[power,featsize] = utils.compute_power_spectrum(natmax,lam,lmax,npoints,nspecies,nnmax,nmax,llmax,lvalues,centers,atom_indexes,all_species,coords,cell,rc,rk,cw,sigma,sg,orthomatrix,sparse_options)
print "Power spectrum computed", time.time()-start 

print "starting feature space dimension", featsize
print "sparsifing power spectrum ...."
start_sparse = time.time()

# Sparsification options.
if sparsefile != '':
    print "We have already sparsified the power spectrum with pre-loaded parameters."
    psparse = power.reshape((2*lam+1)*npoints*natmax,len(power[0,0]))
    featsize = len(psparse[0])
elif ncut > 0:
    # We have not provided a file with sparsification details, but we do want to sparsify, so we'll do it from scratch.
    print "Doing furthest-point sampling sparsification."
    sparsefilename = str(lam)+"_nconf"+str(npoints)+"_sigma"+str(sg)+"_lmax"+str(lmax)+"_nmax"+str(nmax)+"_cutoff"+str(rc)+"_rkill"+str(rk)+"_cweight"+str(cw)+"_init"+str(featsize)
    [psparse,sparse_details] = utils.FPS_sparsify(power.reshape((2*lam+1)*npoints*natmax,featsize),featsize,ncut)
    featsize = len(psparse[0])
    print "Saving sparsification details"
    sparsefilename += "_final"+str(featsize)+".npy"
    if args.outfile != '':
        sparsefilename = args.outfile
    np.save("SPARSE_fps"+sparsefilename,sparse_details[0])
    np.save("SPARSE_Amat"+sparsefilename,sparse_details[1])
else:
    print "Power spectrum will not be sparsified."
    psparse = power.reshape((2*lam+1)*npoints*natmax,featsize)
   
print "done", time.time()-start_sparse 
print "final feature space dimension", featsize

# Reshape power spectrum

if lam==0:
    power = psparse.reshape(npoints,natmax,featsize)
else:
    power = psparse.reshape(npoints,natmax,2*lam+1,featsize)

if lam > 0 and zeta > 1:
    # compute scalar power spectrum
    [power0,featsize0] = utils.compute_power_spectrum(natmax,0,lmax,npoints,nspecies,nnmax,nmax,llmax,lvalues,centers,atom_indexes,all_species,coords,cell,rc,rk,cw,sigma,sg,orthomatrix,sparse_options)
    print "Scalar power spectrum computed"
    print "sparsifing scalar power spectrum ...."
    start_sparse = time.time()
    # SPARSIFY POWER SPECTRUM
    if ncut > 0:
        [psparse,sparse_details] = utils.FPS_sparsify(power0.reshape(npoints*natmax,featsize0),featsize0,ncut)
    else:
        psparse = power0.reshape(npoints*natmax,featsize0)
    featsize0 = len(psparse[0])
    power0 = psparse.reshape(npoints,natmax,featsize0)

# Print power spectrum, if we have not asked for only a sample to be taken (we assume that taking a subset is just for the purpose of generating a sparsification)
if randomnum == -1 and fpsnum == -1:
    print "Saving power spectrum"
    PS_file="PS"+str(lam)+"_nconf"+str(npoints)+"_sigma"+str(sg)+"_lmax"+str(lmax)+"_nmax"+str(nmax)+"_cutoff"+str(rc)+"_rkill"+str(rk)+"_cweight"+str(cw)+"_ncut"+str(ncut)+".npy"
    if args.outfile != '':
        PS_file = args.outfile
    np.save(PS_file,power)
    if lam>0 and zeta>1:
        PS_file="PS"+str(0)+"_nconf"+str(npoints)+"_sigma"+str(sg)+"_lmax"+str(lmax)+"_nmax"+str(nmax)+"_cutoff"+str(rc)+"_rkill"+str(rk)+"_cweight"+str(cw)+"_ncut"+str(ncut)+".npy"
        if args.outfile != '':
            PS_file = "lambda0_"+args.outfile
        np.save(PS_file,power0)

print "Full calculation of power spectrum complete"
