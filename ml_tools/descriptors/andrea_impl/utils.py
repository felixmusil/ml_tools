#!/usr/bin/python

import sys
import numpy as np
from sympy.physics.wigner import wigner_3j
from scipy import special

#############################################################################################

def complex_to_real_transformation(lval):
    """Transformation matrix from complex to real spherical harmonics"""

    if lval==0:
        mat = 1.0

    if lval==1:
        mat = np.array([[1.0j,0.0,1.0j],
                       [0.0,np.sqrt(2.0),0.0],
                       [1.0,0.0,-1.0]
                       ], dtype=complex) / np.sqrt(2.0)

    if lval==2:
        mat = np.array([[1.0j,0.0,0.0,0.0,-1.0j],
                        [0.0,1.0j,0.0,1.0j,0.0],
                        [0.0,0.0,np.sqrt(2.0),0.0,0.0],
                        [0.0,1.0,0.0,-1.0,0.0],
                        [1.0,0.0,0.0,0.0,1.0]      ],dtype=complex) / np.sqrt(2.0)

    if lval==3:
        mat =np.array([[1.0j,0.0,0.0,0.0,0.0,0.0,1.0j],
                       [0.0,1.0j,0.0,0.0,0.0,-1.0j,0.0],
                       [0.0,0.0,1.0j,0.0,1.0j,0.0,0.0],
                       [0.0,0.0,0.0,np.sqrt(2.0),0.0,0.0,0.0],
                       [0.0,0.0,1.0,0.0,-1.0,0.0,0.0],
                       [0.0,1.0,0.0,0.0,0.0,1.0,0.0],
                       [1.0,0.0,0.0,0.0,0.0,0.0,-1.0]   ], dtype=complex) / np.sqrt(2.0)

    if lval==4:
        mat =np.array([[1.0j,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0j],
                       [0.0,1.0j,0.0,0.0,0.0,0.0,0.0,1.0j,0.0],
                       [0.0,0.0,1.0j,0.0,0.0,0.0,-1.0j,0.0,0.0],
                       [0.0,0.0,0.0,1.0j,0.0,1.0j,0.0,0.0,0.0],
                       [0.0,0.0,0.0,0.0,np.sqrt(2.0),0.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,1.0,0.0,-1.0,0.0,0.0,0.0],
                       [0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0],
                       [0.0,1.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0],
                       [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]], dtype=complex) / np.sqrt(2.0)

    return mat

#############################################################################################

def setup_orthomatrix(nmax,rc,rk):

#    sigma = np.zeros(nmax,float)
#    for i in range(nmax):
#        sigma[i] = max(np.sqrt(float(i)),1.0)*(rc)/float(nmax)

    sigma = np.zeros(nmax,float)
    dsigma = 0.2*rc/float(nmax)
    sigma[0] = 0.2*rc/float(nmax)
    for i in xrange(1,nmax):
        sigma[i] = sigma[i-1] + dsigma

    overlap = np.zeros((nmax,nmax),float)
    for n1 in xrange(nmax):
        for n2 in xrange(nmax):
            overlap[n1,n2] = (0.5/(sigma[n1])**2 + 0.5/(sigma[n2])**2)**(-0.5*(3.0 +n1 +n2)) \
                             /(sigma[n1]**n1 * sigma[n2]**n2)*\
                              special.gamma(0.5*(3.0 + n1 + n2))/ (  \
                    (sigma[n1]*sigma[n2])**1.5 * np.sqrt(special.gamma(1.5+n1)*special.gamma(1.5+n2)) )    

    eigenvalues, unitary = np.linalg.eig(overlap)
    sqrteigen = np.sqrt(eigenvalues) 
    diagoverlap = np.diag(sqrteigen)
    newoverlap = np.dot(np.conj(unitary),np.dot(diagoverlap,unitary.T))
    orthomatrix = np.linalg.inv(newoverlap)

    return [orthomatrix,sigma]

##################################################################################################################

def compute_power_spectrum(natmax,lam,lmax,npoints,nspecies,nnmax,nmax,llmax,lvalues,centers,atom_indexes,all_species,coords,cell,rc,rk,cw,sigma,sg,orthomatrix,sparse_options):

    if lam == 0:

        featsize = nspecies*nspecies*nmax**2*(lmax+1)
        if sparse_options[0] != '':
            do_list = sparse_options[1]
            PS = np.zeros((npoints,natmax,len(do_list)),dtype=complex)
        else:
            do_list = xrange(featsize)
            PS = np.zeros((npoints,natmax,featsize),dtype=complex)

        for i in xrange(npoints):
            [nat,omega,harmonic,orthoradint] = fill_arrays(nspecies,nnmax,lmax,nmax,centers,atom_indexes[i],all_species,coords[i],cell[i],rc,rk,cw,sigma,sg,orthomatrix) 
    
            # PRECOMPUTE OMEGA CONJUGATE
            omegatrue = np.zeros((nat,nspecies,nmax,lmax+1,2*lmax+1),complex)
            omegaconj = np.zeros((nat,nspecies,nmax,lmax+1,2*lmax+1),complex)
            for l in xrange(lmax+1):
                for im in xrange(2*l+1):
                    omegatrue[:,:,:,l,im] = omega[:,:,:,l,im]/np.sqrt(np.sqrt(2*l+1))
                    omegaconj[:,:,:,l,im] = np.conj(omega[:,:,:,l,im])/np.sqrt(np.sqrt(2*l+1))
    
            # COMPUTE POWER SPECTRUM
            power = np.einsum('asblm,adnlm->asdbnl',omegatrue,omegaconj)
            power = power.reshape(nat,featsize)
            for iat in xrange(nat):
                inner = np.real(np.dot(power[iat],np.conj(power[iat])))
                power[iat] /= np.sqrt(inner)

            PS[i,:nat] = power[:,do_list]
    
    else:
      
        # PRECOMPUTE WIGNER-3J SYMBOLS
        w3j = np.zeros((2*lam+1,lmax+1,lmax+1,2*lmax+1),float)
        for l1 in xrange(lmax+1):
            for l2 in xrange(lmax+1):
                for m in xrange(2*l1+1):
                    for mu in xrange(2*lam+1):
                        w3j[mu,l1,l2,m] = wigner_3j(lam,l2,l1,mu-lam,m-l1-mu+lam,-m+l1) * (-1.0)**(m-l1)

        featsize = nspecies*nspecies*nmax**2*llmax
        if sparse_options[0] != '':
            do_list = sparse_options[1]
            PS = np.zeros((npoints,natmax,2*lam+1,len(do_list)),dtype=complex)
        else:
            do_list = xrange(featsize)
            PS = np.zeros((npoints,natmax,2*lam+1,featsize),dtype=complex)

        # COMPUTE HARMONIC CONJUGATE, omega conjugate and tensorial power spectrum
        # reduce l dimensionality keeping the nonzero elements
        for i in xrange(npoints):
            [nat,omega,harmonic,orthoradint] = fill_arrays(nspecies,nnmax,lmax,nmax,centers,atom_indexes[i],all_species,coords[i],cell[i],rc,rk,cw,sigma,sg,orthomatrix) 
            power = np.zeros((nat,2*lam+1,nspecies,nspecies,nmax,nmax,lmax+1,lmax+1),complex)
            p2 = np.zeros((nat,2*lam+1,nspecies,nspecies,nmax,nmax,llmax),complex)
            omegaconj = np.zeros((nat,nspecies,2*lam+1,nmax,lmax+1,lmax+1,2*lmax+1),complex)
            harmconj = np.zeros((nat,nspecies,lmax+1,lmax+1,2*lmax+1,2*lam+1,nnmax),dtype=complex)

            for lval in xrange(lmax+1):
                for im in xrange(2*lval+1):
                    for lval2 in xrange(lmax+1):
                        for mu in xrange(2*lam+1):
                            if abs(im-lval-mu+lam) <= lval2:
                                harmconj[:,:,lval2,lval,im,mu,:] = np.conj(harmonic[:,:,lval2,lval2+im-lval-mu+lam,:])
            for iat in xrange(nat):
                for ispe in xrange(nspecies): 
                    omegaconj[iat,ispe] = np.einsum('lnh,lkmvh->vnklm',orthoradint[iat,ispe],harmconj[iat,ispe])
            for iat in xrange(nat):
                for ia in xrange(nspecies):
                    for ib in xrange(nspecies):
                        power[iat,:,ia,ib] = np.einsum('nlv,xmlkv,xlkv->xnmlk',omega[iat,ia],omegaconj[iat,ib],w3j)

            for l in xrange(llmax):
                p2[:,:,:,:,:,:,l] = power[:,:,:,:,:,:,lvalues[l][0],lvalues[l][1]]
            p2 = p2.reshape(nat,2*lam+1,featsize)
            # Normalize power spectrum
            for iat in xrange(nat):
                inner = np.zeros((2*lam+1,2*lam+1),complex)
                for mu in xrange(2*lam+1):
                    for nu in xrange(2*lam+1):
                        inner[mu,nu] = np.dot(p2[iat,mu],np.conj(p2[iat,nu]))
                p2[iat] /= np.sqrt(np.real(np.linalg.norm(inner)))

            PS[i,:nat] = p2[:,:,do_list]

    # Multiply by A_matrix.
    if (sparse_options[0] != ''):
        PS = np.dot(PS,sparse_options[2])

    return [PS,featsize]

##################################################################################################################

def fill_arrays(nspecies,nnmax,lmax,nmax,centers,atom_indexes,all_species,coords,cell,rc,rk,cw,sigma,sg,orthomatrix):

    alpha = 1.0/(2*sg**2)
    sg2 = sg**2

    nat = 0
    for k in centers:
        nat += len(atom_indexes[k])

    length = np.zeros((nat,nspecies,nnmax), dtype=float )
    efact = np.zeros((nat,nspecies,nnmax), dtype=float )
    nneigh = np.zeros((nat,nspecies), dtype=int )

    omega = np.zeros((nat,nspecies,nmax,lmax+1,2*lmax+1),complex)
    harmonic = np.zeros((nat,nspecies,lmax+1,2*lmax+1,nnmax), dtype=complex )
    orthoradint = np.zeros((nat,nspecies,lmax+1,nmax,nnmax),float)
    radint = np.zeros((nat,nspecies,nnmax,lmax+1,nmax),float)
    nat    = 0

    def switch(r):
       if r <= rc-rk:
           f = 1.0
       elif rc-rk < r < rc:
           f = (1.0 + np.cos(np.pi*(r-(rc-rk))/rk))/2.0
       elif r >= rc:
           f = 0.0
       return f

    if np.sum(cell) == 0.0:

        iat = 0
        # Loop over species to center on
        for k in centers:
            nat += len(atom_indexes[k])
            # Loop over centers of that species
            for l in atom_indexes[k]:
                # Loop over all the spcecies to use as neighbours 
                ispe = 0
                for ix in all_species:
                    n=0
                    # Loop over neighbours of that species
                    for m in atom_indexes[ix]:
                        rrm = coords[m] 
                        rr  = rrm - coords[l]
                        # Is it a neighbour within the spherical cutoff ?
                        if np.dot(rr,rr) <= rc**2: 
                            # Central atom?
                            if m == l:
                                length[iat,ispe,n]  = 0.0
                                efact[iat,ispe,n]   = cw
                                harmonic[iat,ispe,0,0,n] = special.sph_harm(0,0,0,0)
                                nneigh[iat,ispe] += 1
                                n += 1
                            else:
                                length[iat,ispe,n]  = np.linalg.norm(rr)
                                theta   = np.arccos(rr[2]/length[iat,ispe,n])
                                phi     = np.arctan2(rr[1],rr[0])
                                efact[iat,ispe,n]   = np.exp(-alpha*length[iat,ispe,n]**2)
                                efact[iat,ispe,n]  *= switch(length[iat,ispe,n])
                                for lval in xrange(lmax+1): 
                                    mxrange = np.arange(-lval,lval+1)
                                    harm = np.conj(special.sph_harm(mxrange,lval,phi,theta))
                                    for im in xrange(2*lval+1):
                                        harmonic[iat,ispe,lval,im,n] = harm[im]
                                nneigh[iat,ispe] += 1
                                n += 1
                    ispe += 1
                iat +=1

    else:

        iat = 0
        # Loop over species to center on
        for k in centers:
            nat += len(atom_indexes[k])
            # Loop over centers of that species
            for l in atom_indexes[k]:
                # Loop over all the spcecies to use as neighbours 
                ispe = 0
                for ix in all_species:
                    n=0
                    # Loop over neighbours of that species
                    for m in atom_indexes[ix]:
                        rrm = coords[m]
                        rr  = rrm - coords[l]
                        # Apply minimum image convention assuming orthorombic cell and rcut < box/2
                        rr[0] -= cell[0,0] * int(round(rr[0] / cell[0,0]))
                        rr[1] -= cell[1,1] * int(round(rr[1] / cell[1,1]))
                        rr[2] -= cell[2,2] * int(round(rr[2] / cell[2,2]))
                        # Is it a neighbour within the spherical cutoff ?
                        if np.dot(rr,rr) <= rc**2:
                            # Central atom?
                            if m == l:
                                length[iat,ispe,n]  = 0.0
                                efact[iat,ispe,n]   = cw
                                harmonic[iat,ispe,0,0,n] = special.sph_harm(0,0,0,0)
                                nneigh[iat,ispe] += 1
                                n += 1
                            else:
                                length[iat,ispe,n]  = np.linalg.norm(rr)
                                theta   = np.arccos(rr[2]/length[iat,ispe,n])
                                phi     = np.arctan2(rr[1],rr[0])
                                efact[iat,ispe,n]   = np.exp(-alpha*length[iat,ispe,n]**2)
                                efact[iat,ispe,n]  *= switch(length[iat,ispe,n])
                                for lval in xrange(lmax+1):
                                    mxrange = np.arange(-lval,lval+1)
                                    harm = np.conj(special.sph_harm(mxrange,lval,phi,theta))
                                    for im in xrange(2*lval+1):
                                        harmonic[iat,ispe,lval,im,n] = harm[im]
                                nneigh[iat,ispe] += 1
                                n += 1
                    ispe += 1
                iat +=1

    for n in xrange(nmax):        
        normfact = np.sqrt(2.0/(special.gamma(1.5+n)*sigma[n]**(3.0+2.0*n)))
        sigmafact = (sg2**2+sg2*sigma[n]**2)/sigma[n]**2
        for l in xrange(lmax+1):
            radint[:,:,:,l,n] = efact[:,:,:] \
                                * 2.0**(-0.5*(1.0+l-n)) \
                                * (1.0/sg2 + 1.0/sigma[n]**2)**(-0.5*(3.0+l+n)) \
                                * special.gamma(0.5*(3.0+l+n))/special.gamma(1.5+l) \
                                * (length[:,:,:]/sg2)**l \
                                * special.hyp1f1(0.5*(3.0+l+n), 1.5+l, 0.5*length[:,:,:]**2/sigmafact)   
        radint[:,:,:,:,n] *= normfact

    for iat in xrange(nat):
        for ispe in xrange(nspecies):
            for neigh in xrange(nneigh[iat,ispe]): 
                for l in xrange(lmax+1):
                    orthoradint[iat,ispe,l,:,neigh] = np.dot(orthomatrix,radint[iat,ispe,neigh,l])

    for iat in xrange(nat):
        for ispe in xrange(nspecies):
            omega[iat,ispe] = np.einsum('lnh,lmh->nlm',orthoradint[iat,ispe],harmonic[iat,ispe])

    return [nat,omega,harmonic,orthoradint]

##################################################################################################################

def sparsify(PS,featsize,ncut):
    """sparsify power spectrum with PCA"""

    eigenvalues, unitary = np.linalg.eigh(np.dot(PS.T,np.conj(PS)))
    psparse = np.dot(PS,unitary[:,featsize-ncut:featsize])

    return psparse 

##################################################################################################################

def FPS_sparsify(PS,featsize,ncut):
    """Sparsify power spectrum with FPS"""

    # Get FPS vector.
    vec_fps = do_fps(PS.T,ncut)
    # Get A matrix.
    C_matr = PS[:,vec_fps]
    UR = np.dot(np.linalg.pinv(C_matr),PS)
    ururt = np.dot(UR,np.conj(UR.T))
    [eigenvals,eigenvecs] = np.linalg.eigh(ururt)
    print "Lowest eigenvalue = %f"%eigenvals[0]
    eigenvals = np.array([np.sqrt(max(eigenvals[i],0)) for i in xrange(len(eigenvals))])
    diagmatr = np.diag(eigenvals)
    A_matrix = np.dot(np.dot(eigenvecs,diagmatr),eigenvecs.T)

    # Sparsify the matrix by taking the requisite columns.
    psparse = np.array([PS.T[i] for i in vec_fps]).T
    psparse = np.dot(psparse,A_matrix)

    # Return the sparsification vector (which we will need for later sparsification) and the A matrix (which we will need for recombination).
    sparse_details = [vec_fps,A_matrix]

    return [psparse,sparse_details]

##################################################################################################################

def do_fps(x, d=0):
    # Code from Giulio Imbalzano

    if d == 0 : d = len(x)
    n = len(x)
    iy = np.zeros(d,int)
    iy[0] = np.random.randint(0,n)
#    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum(x*np.conj(x),axis=1)
    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
    for i in xrange(1,d):
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
        dl = np.minimum(dl,nd)
    return iy

##################################################################################################################
