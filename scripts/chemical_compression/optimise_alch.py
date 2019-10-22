from __future__ import print_function

import argparse
import sys
import pickle
import os
import numpy as np
import conj_grad as cg
import scipy.sparse as sp
import scipy.optimize as opt
from time import time,ctime
##########################################################################################

def p(pdict):

    keys = sorted(pdict.keys())

    def p_inner(i):
        pi = pdict[keys[i]].toarray()
        return pi

    return p_inner

##########################################################################################

def pred(w, u, p, n):

    ''' predicted property '''

    uwut = np.matmul(np.matmul(u, w.transpose((0, 2, 1))), u.T)
    uwut = uwut.transpose((1, 2, 0))
    pred = np.zeros(n)
    for i in xrange(n):
        pred[i] = np.dot(p(i).ravel(), uwut.ravel())

    return pred

##########################################################################################

def loss(w, p, y, nspecies, sigmaw, sigmau):

    d = {'losslast':None, 'ulast':None}

    def loss_inner(u):

        ''' loss function '''

        if np.all(u == d['ulast']): return d['losslast']
        u_unrav = u.reshape((nspecies, -1))
        warray = w(u_unrav)
        diff = y - pred(warray, u_unrav, p, len(y))
        loss = 0.5*(diff**2).sum()
        loss += 0.5*sigmaw*np.linalg.norm(warray)**2
        loss += 0.5*sigmau*np.linalg.norm(u_unrav)**2
        d['losslast'] = loss
        d['ulast'] = u.copy()
        print(loss)

        return loss

    return loss_inner

##########################################################################################

#derivative of the loss w.r.t. its first argument (w)
def grad1(w, p, y, nn, nspecies, nalch, ntrain, sigmaw):

    def grad1_inner(u):
        u_unrav = u.reshape((nspecies, -1))
        warray = w(u_unrav)
        pbar = np.zeros((ntrain, nalch**2, nn))
        ukron = np.kron(u_unrav.T, u_unrav.T)
        diff = y - pred(warray, u_unrav, p, len(y))
        for i in xrange(ntrain):
            pi = p(i).reshape((nspecies, nspecies, nn)).transpose((1, 0, 2))
            pi = pi.reshape((nspecies**2, nn))
            pbar[i] = -np.dot(ukron, pi)*diff[i]
        grad1 = pbar.sum(0)
        grad1 = grad1.reshape((nalch, nalch, nn))
        grad1 = grad1.transpose((2, 0, 1))
        grad1 = grad1.ravel()
        grad1 += sigmaw*warray.ravel()
        return grad1

    return grad1_inner

##########################################################################################

#derivative of the loss w.r.t. its second argument (u)
def grad2(w, p, y, nn, nspecies, nalch, sigmau):

    def grad2_inner(u):
        u_unrav = u.reshape((nspecies, -1))
        warray = w(u_unrav)
        ntrain = len(y)
        diff = y - pred(warray, u_unrav, p, ntrain)
        grad2 = np.zeros((nalch, nspecies))
        wu1 = np.matmul(u_unrav, warray).transpose((0, 2, 1))
        wu2 = np.dot(warray, u_unrav.T)
        wu1 = wu1.transpose((1, 0, 2)).reshape((nalch, nn*nspecies))
        wu2 = wu2.transpose((1, 0, 2)).reshape((nalch, nn*nspecies))
        for i in xrange(ntrain):
            pi1 = p(i).reshape((nspecies, nspecies, nn)).transpose((2, 1, 0))
            pi2 = pi1.transpose((0, 2, 1))
            pi1 = pi1.reshape((nn*nspecies, nspecies))
            pi2 = pi2.reshape((nn*nspecies, nspecies))
            grad2 -= diff[i]*(np.dot(wu1, pi1) + np.dot(wu2, pi2))
        grad2 = grad2.T
        grad2 += sigmau*u_unrav
        return grad2

    return grad2_inner

##########################################################################################

#derivative of the loss w.r.t. its first argument (w) twice (the Hessian)
def grad11(p, nn, nspecies, nalch, ntrain, sigmaw):

    def grad11_inner(u):
        u_unrav = u.reshape((nspecies, -1))
        pbar = np.zeros((ntrain, nalch**2, nn))
        ukron = np.kron(u_unrav.T, u_unrav.T)
        for i in xrange(ntrain):
            pi = p(i).reshape((nspecies, nspecies, nn)).transpose((1, 0, 2))
            pi = pi.reshape((nspecies**2, nn))
            pbar[i] = np.dot(ukron, pi)
        pbar = pbar.transpose((0, 2, 1))
        pbar = pbar.reshape((ntrain, nn*nalch**2)).T
        grad11 = np.dot(pbar, pbar.T)
        grad11 += sigmaw*np.eye(nalch**2*nn)
        return grad11

    return grad11_inner

##########################################################################################

#derivative of the loss w.r.t. its first (w) and second argument (u)
def grad12(w, p, y, nn, nspecies, nalch):

    def grad12_inner(u):
        u_unrav = u.reshape((nspecies, -1))
        warray = w(u_unrav)
        diff = y - pred(warray, u_unrav, p, len(y))
        ntrain = len(y)
        ukron = np.kron(u_unrav.T, u_unrav.T)
        grad12 = np.zeros((nn, nalch, nalch, nspecies, nalch))
        grad12a = np.zeros((nalch, nspecies))
        wu1 = np.matmul(u_unrav, warray).transpose((0, 2, 1))
        wu2 = np.dot(warray, u_unrav.T)
        wu1 = wu1.transpose((1, 0, 2)).reshape((nalch, nn*nspecies))
        wu2 = wu2.transpose((1, 0, 2)).reshape((nalch, nn*nspecies))
        for l in xrange(ntrain):
            pl1 = p(l).reshape((nspecies, nspecies, nn)).transpose((2, 1, 0))
            pl2 = pl1.transpose((0, 2, 1))
            vec0 = np.matmul(ukron, pl1.transpose((1, 2, 0)).reshape((nspecies**2, nn)))
            vec0 = vec0.T.reshape((nn, nalch, nalch))
            vec1 = pl1.reshape((nn*nspecies, nspecies))
            vec2 = pl2.reshape((nn*nspecies, nspecies))
            grad12a = (np.dot(wu1, vec1) + np.dot(wu2, vec2)).T
            grad12 += np.multiply.outer(vec0, grad12a)
            plu1 = np.matmul(pl1, u_unrav)*diff[l]
            plu2 = np.matmul(pl2, u_unrav)*diff[l]
            for i in xrange(nalch):
                for j in xrange(nalch):
                    grad12[:, i, j, :, j] -= plu2[:, :, i] 
                    grad12[:, i, j, :, i] -= plu1[:, :, j] 
        grad12 = grad12.reshape(nn*nalch**2, nspecies*nalch)
        return grad12

    return grad12_inner

##########################################################################################

#derivative of w w.r.t. u
def dwdu(grad11, grad12):

    d = {'dwdulast': None, 'ulast': None}

    def dwdu_inner(u):
        if np.all(u == d['ulast']): return d['dwdulast']
        inv = np.linalg.pinv(grad11(u))
        dwdu = -np.dot(inv, grad12(u))
        d['dwdulast'] = dwdu
        d['ulast'] = u.copy()
        return dwdu

    return dwdu_inner

##########################################################################################

#total derivative of the loss w.r.t. u
def grad_u(grad1, grad2, dwdu):

    def grad_u_inner(u):
        term1 = grad2(u)
        term2 = np.dot(grad1(u), dwdu(u)).reshape(term1.shape)
        grad_u = term1 + term2
        return grad_u.ravel()

    return grad_u_inner

##########################################################################################

#w as a function of u
def wfun(p, y, nn, nspecies, nalch, ntrain, sigmaw):

    d = {'wlast': None, 'ulast': None}

    def w_inner(u):
        if np.all(u == d['ulast']): return d['wlast']
        u_unrav = u.reshape((nspecies, -1))
        pbar = np.zeros((ntrain, nalch**2, nn))
        ukron = np.kron(u_unrav.T, u_unrav.T)
        for i in xrange(ntrain):
            pi = p(i).reshape((nspecies, nspecies, nn)).transpose((1, 0, 2))
            pi = pi.reshape((nspecies**2, nn))
            pbar[i] = np.dot(ukron, pi)
        pbar = pbar.transpose((0, 2, 1))
        pbar = pbar.reshape((ntrain, nn*nalch**2)).T
        w, norm = cg.cg(pbar, np.dot(pbar, y), np.zeros((nn*nalch**2)), lam=sigmaw)
        w = w.reshape((nn, nalch, nalch))
        d['wlast'] = w
        d['ulast'] = u.copy()
#        cov = np.dot(pbar, pbar.T) + sigmaw*np.eye(nalch**2*nn)
#        inv = np.linalg.pinv(cov)
#        w2 = np.dot(inv, np.dot(pbar, y))
        return w

    return w_inner

##########################################################################################

def prepare_input(ffing, fprop, ntrain):

    ''' load the fingerprints and properties '''

    try:
        f = open(ffing, 'rb')
    except:
        print("Cannot find the fingerprint file")
        sys.exit()

    p = pickle.load(f)
    if ntrain == 0: ntrain = len(p)
    p = dict((k, j) for k, j in enumerate(p.values()) if k < ntrain)
    f.close()

    try:
        y = np.load(fprop)
    except:
        print("Cannot find the property file")
        sys.exit()

    y = y[:ntrain] - y[:ntrain].mean()

    return ntrain, p, y

##########################################################################################

def prepare_u(fu, species, nn, nspecies, nalch):

    ''' initialise u '''

    if fu != '':

        try:
            u = np.load(fu).T
        except:
            print("Cannot find the u data")
            sys.exit()

    else:

        #Pauling electronegativities
        enev_dict = {1:2.20, 2:0.0, 3:0.98, 4:1.57, 5:2.04, 6:2.55, 7:3.04, 8:3.44, \
                     9:3.98, 10:0.0, 11:0.93, 12:1.31, 13:1.61, 14:1.90, 15:2.19, \
                     16:2.58, 17:3.16, 18:0.0, 19:0.82, 20:1.00, 21:1.36, 22:1.54, \
                     23:1.63, 24:1.66, 25:1.55, 26:1.83, 27:1.88, 28:1.91, 29:1.90, \
                     30:1.65, 31:1.81, 32:2.01, 33:2.18, 34:2.55, 35:2.96, 36:3.00, \
                     37:0.82, 38:0.95, 39:1.22, 40:1.33, 41:1.6, 42:2.16, 43:1.9, \
                     44:2.2, 45:2.28, 46:2.20, 47:1.93, 48:1.69, 49:1.78, 50:1.96, \
                     51:2.05, 52:2.1, 53:2.66, 54:2.6, 55:0.79, 56:0.89, 57:1.10, \
                     58:1.12, 59:1.13, 60:1.14, 61:0.0, 62:1.17, 63:0.0, 64:1.20, \
                     65:0.0, 66:1.22, 67:1.23, 68:1.24, 69:1.25, 70:0.0, 71:1.27, \
                     72:1.3, 73:1.5, 74:2.36, 75:1.9, 76:2.2, 77:2.20, 78:2.28, \
                     79:2.54, 80:2.00, 81:1.62, 82:2.33, 83:2.02, 84:2.0, 85:2.2, \
                     86:0.0, 87:0.0, 88:0.9, 89:1.1, 90:1.3, 91:1.5, 92:1.38, 93:1.36, \
                     94:1.28, 95:1.3, 96:1.3, 97:1.3, 98:1.3, 99:1.3, 100:1.3, \
                     101:1.3, 102:1.3}

        #van der Waals radii
        # Van der Wals radii from W. M. Haynes, Handbook of Chemistry and Physics 95th Edition, CRC Press, New York, 2014, ISBN-10: 1482208679, ISBN-13: 978-1482208672.
        # in Angtrom
        vdw_dict = {1: 1.1, 2: 1.4, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.7, 7: 1.55,
                8: 1.52, 9: 1.47, 10: 1.54, 11: 2.27, 12: 1.73, 13: 1.84,
                14: 2.1, 15: 1.8, 16: 1.8, 17: 1.75, 18: 1.88, 19: 2.75,
                20: 2.31, 21: 2.15, 22: 2.11, 23: 2.07, 24: 2.06, 25: 2.05,
                26: 2.04, 27: 2.0, 28: 1.97, 29: 1.96, 30: 2.01, 31: 1.87,
                32: 2.11, 33: 1.85, 34: 1.9, 35: 1.85, 36: 2.02, 37: 3.03,
                38: 2.49, 39: 2.32, 40: 2.23, 41: 2.18, 42: 2.17, 43: 2.16,
                44: 2.13, 45: 2.1, 46: 2.1, 47: 2.11, 48: 2.18, 49: 1.93,
                50: 2.17, 51: 2.06, 52: 2.06, 53: 1.98, 54: 2.16, 55: 3.43,
                56: 2.68, 57: 2.43, 58: 2.42, 59: 2.4, 60: 2.39, 62: 2.36,
                63: 2.35, 64: 2.34, 65: 2.33, 66: 2.31, 67: 2.3, 68: 2.29,
                69: 2.27, 70: 2.26, 71: 2.24, 72: 2.23, 73: 2.22, 74: 2.18,
                75: 2.16, 76: 2.16, 77: 2.13, 78: 2.13, 79: 2.14, 80: 2.23,
                81: 1.96, 82: 2.02, 83: 2.07}


        enevs = np.zeros((nspecies))
        vdw = np.zeros((nspecies))
        for i in xrange(nspecies):
            enevs[i] = enev_dict[species[i]] 
            vdw[i] = vdw_dict[species[i]] 

        kappa = np.zeros((nspecies, nspecies))
	s1 = 0.5
	s2 = 0.5
        for i in xrange(nspecies):
            for j in xrange(i, nspecies):
                kappa[i, j] = np.exp(-0.5*(enevs[i] - enevs[j])**2/s1**2)
                kappa[i, j] *= np.exp(-0.5*(vdw[i] - vdw[j])**2/s2**2)
                kappa[j, i] = kappa[i, j]
        u, v = np.linalg.eigh(kappa)
        u = np.sqrt(abs(u[-nalch:]))
        u = np.dot(v[:, -nalch:], np.diag(u))

    return u

##########################################################################################
##########################################################################################

def main(ffing, fprop, seed, nalch, sigmaw, sigmau, suffix, fu, species, \
         ntrain, nmax, lmax):
    
    print('Start : {}'.format(ctime()))
    sys.stdout.flush()
    ffing = str(ffing)
    fprop = str(fprop)
    seed = int(seed)
    nalch = int(nalch)
    sigmaw = float(sigmaw)
    sigmau = float(sigmau)
    suffix = str(suffix)
    fu = str(fu)
    species = sorted([int(species) for species in species.split(',')])
    nspecies = len(species)
    ntrain = int(ntrain)
    np.random.seed(seed)
    nn = nmax**2*(lmax + 1)

    ntrain, pdict, y = prepare_input(ffing, fprop, ntrain)
    u = prepare_u(fu, species, nn, nspecies, nalch)

    nalchstr = 'nalch'+str(nalch)
    #if suffix != '': suffix = '_'+suffix

    #shuffling to undo fps
    rr = range(ntrain)
    np.random.shuffle(rr)
    pdict2 = {}
    for i, j in enumerate(rr):
        pdict2[i] = pdict[j]
    pdict = pdict2
    y = y[:ntrain]
    y = y[rr]
    
    #separate the fingerprints across the data set into A and B
    plist = [(key, val) for (key, val) in pdict.iteritems()]
    p1 = dict(plist[:ntrain/2])
    p2 = dict(plist[ntrain/2:])
    y1 = y[:ntrain/2]
    y2 = y[ntrain/2:]
    p1 = p(p1)
    p2 = p(p2)

    #initialise all the functions for AB <-> BA crossval opt
    wa = wfun(p1, y1, nn, nspecies, nalch, len(y1), sigmaw)
    wb = wfun(p2, y2, nn, nspecies, nalch, len(y2), sigmaw)
    g1a = grad1(wb, p1, y1, nn, nspecies, nalch, len(y1), sigmaw=0.0)
    g1b = grad1(wa, p2, y2, nn, nspecies, nalch, len(y2), sigmaw=0.0)
    g2a = grad2(wb, p1, y1, nn, nspecies, nalch, sigmau=0.0)
    g2b = grad2(wa, p2, y2, nn, nspecies, nalch, sigmau=0.0)
    g11a = grad11(p1, nn, nspecies, nalch, len(y1), sigmaw)
    g11b = grad11(p2, nn, nspecies, nalch, len(y2), sigmaw)
    g12a = grad12(wa, p1, y1, nn, nspecies, nalch)
    g12b = grad12(wb, p2, y2, nn, nspecies, nalch)
    dwabydu = dwdu(g11a, g12a)
    dwbbydu = dwdu(g11b, g12b)
    #total derivative w.r.t. u of the test loss on set A
    gua = grad_u(g1a, g2a, dwbbydu)
    #total derivative w.r.t. u of the test loss on set B
    gub = grad_u(g1b, g2b, dwabydu)
    #test loss on set A
    fa = loss(wb, p1, y1, nspecies, sigmaw=0.0, sigmau=0.0)
    #test loss on set B
    fb = loss(wa, p2, y2, nspecies, sigmaw=0.0, sigmau=0.0)

    #construct the total loss (A + B)
    def f(u): return fa(u) + fb(u) + 0.5*sigmau*np.linalg.norm(u)**2
    #construct the total gradient of the loss w.r.t. u (A + B)
    def df(u): return gua(u) + gub(u) + sigmau*u.ravel()

    print('Start opt: {}'.format(ctime()))
    for i in xrange(50):
        if os.path.exists('EXIT') == True: 
            print("Found EXIT file")
            break
        iterstr = '-niter'+str(i)
        np.save(suffix+iterstr+'-u_mat+'+'.npy', u.T)
        np.save(suffix+iterstr+'-kappa_mat'+'.npy', np.dot(u, u.T))
        print("RMSE %.6f" % np.sqrt(2.0*(f(u) - 0.5*sigmau*np.linalg.norm(u)**2)/float(ntrain)))
        res = opt.minimize(fun=f, x0=u, jac=df, method='L-BFGS-B', \
                           options={'maxiter':10, 'disp':True})
        u = res.x.reshape((nspecies, nalch))
        sys.stdout.flush()

    iterstr = '-niter'+str(i+1)
    np.save(suffix+iterstr+'-u_mat+'+'.npy', u.T)
    np.save(suffix+iterstr+'-kappa_mat'+'.npy', np.dot(u, u.T))
    print('dump results in: {} {}'.format(suffix+iterstr+'-u_mat+'+'.npy',suffix+iterstr+'-kappa_mat+'+'.npy'))
    print('Finished: {}'.format(ctime()))
##########################################################################################
##########################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ffing', type=str, help='Fingerprint file')
    parser.add_argument('-fprop', type=str, help='Property file')
    parser.add_argument('-species', type=str, help='List of elements e.g. "1 2 3"')
    parser.add_argument('-nmax', type=int, default=9, help='Maximum radial label')
    parser.add_argument('-lmax', type=int, default=9, help='Maximum angular label')
    parser.add_argument('--seed', type=int, default=1, help='Seed for RNG')
    parser.add_argument('--nalch', type=int, default=1, help='Size of alchemical basis')
    parser.add_argument('--sigmaw', type=float, default=1.0e-6, help='Sigma for w')
    parser.add_argument('--sigmau', type=float, default=1.0e-6, help='Sigma for u')
    parser.add_argument('--suffix', type=str, default='', help='Filename suffix')
    parser.add_argument('--fu', type=str, default='', help='Location of the u data file')
    parser.add_argument('--ntrain', type=int, default=0, help='Number of training structures')
    args = parser.parse_args()

    main(args.ffing, args.fprop, args.seed, args.nalch, args.sigmaw, args.sigmau, \
         args.suffix, args.fu, args.species, args.ntrain, args.nmax, args.lmax)
