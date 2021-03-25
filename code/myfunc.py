"""
Various functions used in the solver of Abel integral inversions.
Functions
-------
    conjugate_grad : conjugate gradient method to solve linear problem Ax = b
    power          : 
@author: Cecile Della Valle
@date: 03/01/2021
"""
# global import
import math
import numpy as np
# local import
from code.fracpower import Adaptative_Quad_DE

def conjugate_grad(A,b,x=None,eps=1e-16):
    """
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
        A     (numpy.array): positive semi-definite (symmetric) matrix size n,n
        b     (numpy.array): vector size n
        x     (numpy.array): vector of initial point size n
    Returns
    -------
              (numpy.array): solution x such that Ax = b
    """
    n = b.shape[0]
    if not x:
        xk = b.copy()
    loss = []
    # initialisayion
    gk = np.dot(A, xk) - b
    pk = - gk
    gk_norm = np.dot(gk, gk)
    # loop
    for i in range(2*n):
        Apk = np.dot(A, pk)
        alpha = gk_norm / np.dot(pk, Apk)
        xk += alpha * pk
        gk += alpha * Apk
        gkplus1_norm = np.dot(gk,gk)
        beta = gkplus1_norm / gk_norm
        gk_norm = gkplus1_norm
        if gkplus1_norm < eps:
            break
        pk = beta * pk - gk
        loss.append(gkplus1_norm)
    return xk,loss

def Power(M,r):
    """
    Return a matrix M of fractional power r,
    M is not necessary semi-definite symmetric.
    If M is not symmetric, M need to be invertible, otherwize an exception is raised.
    Parameters
    ----------
        M     (numpy.array):  matrix size (n,n), either symmetric or invertible
    Returns
    -------
              (numpy.array): M to power r, size (n,n)
    """
    test = np.linalg.norm(np.transpose(M) - M)/np.linalg.norm(M)
    if test < 0.01 : # M is symmetric
        D,P = eigh(M)
        D = np.diag(D**r)
        D = P.dot(D).dot(np.transpose(P))
    else: # M is not symmetric, but it has to be invertible
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print('The matrix {} is not invertible'.format{M})
        D = Adaptative_Quad_DE(M,Minv,r,eps=10**-5,niter=5)
    return D
        
def Export(x,y,folder,name):
    """
    Export a set of point (x,y) in the folder 'folder' with the name 'name'.
    Parameters
    ----------
        x     (numpy.array): vector point size n
        y     (numpy.array): vector point size n
        folder        (str): folder name
        name          (str): name of file
    Returns
    -------
              -
    """
    
    # initialisation
    Npoint = np.size(x)
    with open(folder+'/'+name+'.txt', 'w') as f:
        f.writelines('xdata ydata \n')
        for i in range(Npoint):
            web_browsers = ['{0}'.format(x[i]),' ','{0} \n'.format(y[i])]
            f.writelines(web_browsers)