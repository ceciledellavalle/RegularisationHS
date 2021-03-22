import numpy as np
# global import
import math
import numpy as np

def conjugate_grad(A,b,x=None,eps=1e-16):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
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
