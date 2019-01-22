#!/usr/bin/python2

import numpy as np
from ase.io import read 
from ase.geometry import find_mic
from ge import gaussian_expansion as ge

########################################################################################## 

def cutoff(r, rcut):
  ctwidth = 0.5
  if r <= rcut - ctwidth: cutoff = 1.0
  elif r >= rcut: cutoff = 0.0
  else: cutoff = 0.5*(1.0 + np.cos(np.pi*(r - rcut + ctwidth)/ctwidth))
  return cutoff

########################################################################################## 

def density(nmax, lmax, brcut, sigma):
  #Gauss-Legendre quadrature points and weights
  x, w = ge.gaulegf(x1=0.0, x2=brcut, n=nmax+1)

  def inner(rij, cost, phi, lmax=lmax):
    #rij interatomic distance
    #cost cos of theta
    #lmax maximum spherical harmonic l
    f1 = np.zeros((nmax+1, lmax+1))
    for i in xrange(nmax+1):
      f1[i, :] = ge.fsubquad(r2=x[i], sigma2=sigma, rij2=rij, lmax=lmax)
      f1[i, :] *= np.sqrt(w[i])
    f2 = np.zeros((nmax+1, lmax+1, lmax+1))
    f2 = ge.f2sub(f1, cost=cost)
    f2[:, :, 1:] *= np.sqrt(2.0)
    eiphi = np.cos(phi) + 1.0j*np.sin(phi)
    eiphiar = np.array([eiphi**i for i in xrange(lmax+1)])
    f2 = f2.astype(complex)
    f2 *= eiphiar
    return f2

  return inner

##########################################################################################

def power_spectrum(nmax, lmax, f1, f2):
  ps = np.zeros((nmax+1, nmax+1, lmax+1), dtype=complex)
  f2conj = np.conj(f2)
  ps = np.einsum('ikl,jkl -> ijk', f1, f2conj)
  ps /= np.array([np.sqrt(2.0*l + 1.0) for l in xrange(lmax+1)])
  ps = np.real(ps).ravel()
  return ps

##########################################################################################

def rconvert(r):
  rij = np.sqrt(np.dot(r, r))
  if rij > 0.0:
    cost = r[2]/rij
  else:
    cost = 0.0
  phi = np.arctan2(r[1], r[0])
  return rij, cost, phi

##########################################################################################

def get_descriptor(centres, xyz, species, nmax, lmax, rcut, gdens):
  coords = xyz.get_positions()
  ans = xyz.get_atomic_numbers()
  nspecies = len(species)
  common = [species.index(j) for j in sorted(list(set(ans)))]
  common = zip(common, sorted(list(set(ans))))
  cell = xyz.get_cell()
  if cell.sum() == 0.0: pbc = False
  else: pbc = xyz.get_pbc()
  centind = [i for i, j in enumerate(ans) if j in centres]
  desclist = []
  for l, centre in enumerate(centind):
    f = np.zeros((nspecies, nmax+1, lmax+1, lmax+1), dtype=complex) 
    dr = find_mic(coords - coords[centre], cell=cell, pbc=pbc)[0]
    for i, spec in common:
      labels = np.where(ans == spec)[0]
      for j in labels:
        rij, cost, phi = rconvert(dr[j])
        if rij >= rcut: continue
        f[i] += gdens(rij, cost, phi)*cutoff(rij, rcut)
    desc = np.zeros((nspecies*(nspecies+1)/2, (nmax+1)**2*(lmax+1)))
    counter = 0
    for i in xrange(nspecies):
      for j in xrange(i, nspecies):
        if i == j: mult = 1.0
        else: mult = np.sqrt(2.0)
        desc[counter] = mult*power_spectrum(nmax, lmax, f[i], f[j])
        counter += 1
    desc = desc.ravel()
#    desc = np.zeros((nspecies, nspecies, (nmax+1)**2*(lmax+1)))
#    counter = 0
#    for i in xrange(nspecies):
#      for j in xrange(nspecies):
#        desc[i, j] = power_spectrum(nmax, lmax, f[i], f[j])
#    desc = desc.ravel()
    desclist += [desc]
  return desclist

##########################################################################################
