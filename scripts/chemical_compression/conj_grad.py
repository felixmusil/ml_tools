''' conjugate gradient code to solve (A + lam) x = b, where A = phi phi.T '''

import numpy as np

np.seterr(all='raise')

def cg(phi, b, x0, lam=1.0, maxiter=1000, tol=1.0e-25):

    n1 = phi.shape[0]
    n2 = phi.shape[1]

    r = np.zeros((2, n1))
    x = np.zeros((2, n1))
    p = np.zeros((2, n1))
    alpha = np.zeros((2))
    beta = np.zeros((2))
    norm = np.zeros((2))
    
    x[0] = x0
    ax0 = np.dot(phi, np.dot(phi.T, x[0]))
    ax0 += lam*x[0]
    r[0] = b - ax0
    p[0] = r[0]
    norm[0] = np.linalg.norm(r[0])**2

    for i in xrange(maxiter):

        lab1 = i%2
        lab2 = (i+1)%2

        pkphi = np.dot(p[lab1], phi)
        alpha[lab1] = norm[lab1]
        try:
            alpha[lab1] /= np.dot(pkphi, pkphi) + lam*np.dot(p[lab1], p[lab1])
            x[lab2] = x[lab1] + alpha[lab1]*p[lab1]
            r[lab2] = r[lab1] - alpha[lab1]*np.dot(phi, np.dot(phi.T, p[lab1])) 
            r[lab2] -= alpha[lab1]*lam*p[lab1]
            norm[lab2] = np.linalg.norm(r[lab2])**2
            beta[lab1] = norm[lab2]/norm[lab1]
            p[lab2] = r[lab2] + beta[lab1]*p[lab1]
        except:
            x[lab2] = x[lab1]
            norm[lab2] = norm[lab1]
            break

        if norm[lab2] <= tol: break

    return x[lab2], norm[lab2]
