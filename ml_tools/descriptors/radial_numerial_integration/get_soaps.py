

import argparse
import numpy as np
import env_reader as er
from .ge import gaussian_expansion as ge
from ase.io import read

##########################################################################################

def get_soaps(centres, species, nmax, lmax, rc, gdens):
  nspecies = len(species)

  def inner(frames):
    n = len(frames)
    soaps_list = []
    for i in xrange(n):
      soap = er.get_descriptor(centres, frames[i], species, nmax, lmax, rc, gdens)
      soap = np.vstack(soap)
      for k in xrange(soap.shape[0]):
        soap[k] /= np.linalg.norm(soap[k])
      soaps_list += [soap]
    return np.vstack(soaps_list)

  return inner

##########################################################################################
##########################################################################################

def main(suffix, fxyz, rc, species, nmax, lmax, awidth, nframes, centres):

  suffix = str(suffix)
  if suffix != '': suffix = '_'+suffix
  fxyz = str(fxyz)
  if centres == '': centres = species
  species = sorted([int(species) for species in species.split()])
  centres = sorted([int(element) for element in centres.split()])
  nmax = int(nmax)
  lmax = int(lmax)
  awidth = float(awidth)
  nframes = int(nframes)
  if nframes == 0: nframes = ''
  frames = read(fxyz, ':'+str(nframes))
  nframes = len(frames)

  brcut = rc + 3.0*awidth
  gdens = er.density(nmax, lmax, brcut, awidth)
  gsoaps = get_soaps(centres, species, nmax, lmax, rc, gdens)
  soaps = gsoaps(frames)
  np.save('p'+suffix+'.npy', soaps)

##########################################################################################
##########################################################################################

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-fxyz', type=str, help='Location of xyz file')
  parser.add_argument('-species', type=str, help='List of elements e.g. "1 2 3"')
  parser.add_argument('--suffix', type=str, default='', help='Filename suffix')
  parser.add_argument('--rc', type=float, default=3.0, help='Cutoff radius')
  parser.add_argument('--nmax', type=int, default=9, help='Maximum radial label')
  parser.add_argument('--lmax', type=int, default=9, help='Maximum angular label')
  parser.add_argument('--awidth', type=float, default=0.3, help='Atom width')
  parser.add_argument('--nframes', type=int, default=0, help='Number of frames')
  parser.add_argument('--centres', type=str, default='', help='List of elements to centre on')
  args = parser.parse_args()

  main(args.suffix, args.fxyz, args.rc, args.species, args.nmax, args.lmax, \
       args.awidth, args.nframes, args.centres)
