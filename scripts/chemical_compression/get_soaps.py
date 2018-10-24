
import numpy as np
import quippy as qp
import re
import argparse
import sys
import pickle
from scipy import sparse as sp
from string import Template

##########################################################################################

def order_soap(soap, species, nspecies, nab, subspecies, nsubspecies, nsubab, nmax, lmax, nn):

    p = np.zeros((nsubspecies, nsubspecies, nn))

    #translate the fingerprints from QUIP
    counter = 0
    p = np.zeros((nsubspecies, nsubspecies, nmax, nmax, lmax + 1))
    rs_index = [(i%nmax, (i - i%nmax)/nmax) for i in xrange(nmax*nsubspecies)]
    for i in xrange(nmax*nsubspecies):
        for j in xrange(i + 1):
            if i != j: mult = np.sqrt(0.5)
            else: mult = 1.0
            for k in xrange(lmax + 1):
                n1, s1 = rs_index[i]
                n2, s2 = rs_index[j]
                p[s1, s2, n1, n2, k] = soap[counter]*mult
                if s1 == s2: p[s1, s2, n2, n1, k] = soap[counter]*mult
                counter += 1
    for s1 in xrange(nsubspecies):
        for s2 in xrange(s1):
            p[s2, s1] = p[s1, s2].transpose((1, 0, 2))

    p = p.reshape((nsubspecies, nsubspecies, nn))
    p_full = np.zeros((nspecies, nspecies, nn))
    indices = [species.index(i) for i in subspecies]
    for i, j in enumerate(indices):
        for k, l in enumerate(indices):
            p_full[j, l] = p[i, k]
    p = p_full.reshape((nspecies**2, nn))

    return p

##########################################################################################

def get_soaps(soapstr, rc, species, nspecies, nmax, lmax, nn, nab):

    def inner(frames):
        soaps_list = []
        n = len(frames)
        for i in xrange(n):
            frame = frames[i]
            frame.set_cutoff(rc)
            frame.calc_connect()
            subspecies = sorted(list(set([atom.number for atom in frame if atom.number in species])))
            nsubspecies = len(subspecies)
            nsubab = nsubspecies*(nsubspecies + 1)/2
            speciesstr = '{'+re.sub('[\[,\]]', '', str(subspecies))+'}'
            soapstr2 = soapstr.substitute(nspecies=nsubspecies, ncentres=nsubspecies, \
                                          species=speciesstr, centres=speciesstr)
            desc = qp.descriptors.Descriptor(soapstr2)
            soap = desc.calc(frame, grad=False)['descriptor']
            nenv = soap.shape[0]
            soaps = np.zeros((nenv, nspecies**2, nn))
            for j in xrange(nenv):
                soaps[j] = order_soap(soap[j], species, nspecies, nab, subspecies, \
                                      nsubspecies, nsubab, nmax, lmax, nn)
            soaps_list.append(sp.csc_matrix(soaps.mean(axis=0)))
        return soaps_list

    return inner

##########################################################################################
##########################################################################################

def main(suffix, fxyz, rc, species, nmax, lmax, awidth, nframes, cutoff_dexp, cutoff_scale):

    suffix = str(suffix)
    if suffix != '': suffix = '_'+suffix
    fxyz = str(fxyz)
    cutoff = float(rc)
    species = sorted([int(species) for species in species.split()])
    nmax = int(nmax)
    lmax = int(lmax)
    awidth = float(awidth)
    nframes = int(nframes)
    if nframes == 0: nframes = None
    nspecies = len(species)
    nn = nmax**2*(lmax + 1)
    nab = nspecies*(nspecies+1)/2
    cutoff_dexp = int(cutoff_dexp)
    cutoff_scale = float(cutoff_scale)

    frames = qp.AtomsList(fxyz, stop=nframes)
    nframes = len(frames)

    soapstr = Template('average=F normalise=T soap cutoff_dexp=$cutoff_dexp \
                        cutoff_scale=$cutoff_scale central_reference_all_species=F \
                        central_weight=1.0 covariance_sigma0=0.0 atom_sigma=$awidth \
                        cutoff=$rc cutoff_transition_width=0.5 n_max=$nmax l_max=$lmax \
                        n_species=$nspecies species_Z=$species n_Z=$ncentres Z=$centres')
    soapstr = soapstr.safe_substitute(rc=rc, nmax=nmax, lmax=lmax, awidth=awidth, \
                                      cutoff_dexp=cutoff_dexp, cutoff_scale=cutoff_scale)
    soapstr = Template(soapstr)

    gsoaps = get_soaps(soapstr, rc, species, nspecies, nmax, lmax, nn, nab)
    soaps_list = gsoaps(frames)

    p = {}
    for i in xrange(nframes):
        p[i] = soaps_list[i]
    
    f = open(suffix+'.pckl', 'wb')
    pickle.dump(p, f)
    f.close()

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
    parser.add_argument('--cutoff_dexp', type=int, default=0, help='Witch\'s exponent')
    parser.add_argument('--cutoff_scale', type=float, default=1.0, help='Witch\'s scale')
    args = parser.parse_args()

    main(args.suffix, args.fxyz, args.rc, args.species, args.nmax, args.lmax, \
         args.awidth, args.nframes, args.cutoff_dexp, args.cutoff_scale)
