"""
Various functions used in the solver of Abel integral inversions.
Functions
-------
    MatrixGen      : compute operator, symmetrical operator and derivative
    Conjugate_grad : conjugate gradient method to solve linear problem Ax = b
    Power          : compute fractional power of matricees
@author: Cecile Della Valle
@date: 03/01/2021
"""
# global import
import math
import numpy as np
# local import
from code.fracpower import Adaptative_Quad_DE
from scipy.linalg import inv,pinvh,eig,eigh

# Generate the discretized operator T and D
def MatrixGen(a,p,nx,kernel,method1,method2):
    """
      Compute the operators linked to the inverse problem.
         Parameters
         ----------
              a          (float): order of ill-posedness
              p          (float): order of regularization
              nx           (int): discretization time step
              kernel      (bool): True if the Abel operator includes a kernel k(t,s)= 1/(t+s)^0.5, 
                                  False otherwise, and k(t,s)=1
              method1        (str): 'trapeze' for the trapeze approximation of T, 
                                   is also implemented the finite element method ('eltP0') 
                                   and the computation using semi-groups properties ('fracpower')
              method2        (str): 'explicit' for the finite difference method to approximate B,
                                   but can also be define as (tTT)^-1 ('inv')
        Retruns
        ----------
             (numpy.array): Abel operator T, size nx,nx
             (numpy.array): symmetric of Abel operator tTT, size nx,nx
             (numpy.array): symmetric regularisation operator tDD, size nx,nx
    """
    # ==================================================
    # Matrice opérateur
    # method 0 : kernel (trapeze)
    if kernel==True:
        T = np.zeros((nx,nx))
        coeff = 1/(2*a)*nx**-a
        for i in range(nx):
            for j in range(i):#lower half
                if (j==0):
                     T[i,j] = coeff*1/np.sqrt(i)*(i**a-(i-1)**a)
                elif i==j:#diagonal
                    T[i,j] = coeff*1/2*1/np.sqrt(2*i)
                else:
                    T[i,j] = 1/np.sqrt(i+j)*coeff*((i-j+1)**a\
                                -(i-j-1)**a)
    # methode 1 : trapeze
    elif method1=='trapeze': #default
        T = np.zeros((nx,nx))
        coeff = 1/(2*a)*nx**-a
        for i in range(1,nx):
            for j in range(i):#lower half
                if j==0: 
                    T[i,j] = coeff*(i**a-(i-1)**a)
                elif i==j:#diagonal
                    T[i,j] = coeff
                else:
                    T[i,j] = coeff*((i-j+1)**(a)\
                                -(i-j-1)**(a))
    # methode 2 : element fini P0
    elif method1=='eltP0':
        T = np.zeros((nx,nx))
        coeff = 1/(a*(a+1))*nx**-a
        for i in range(nx):
            for j in range(nx):#lower half
                if i<j:
                   T[i,j] = coeff*((j-i+1)**(a+1)\
                                   +(j-i-1)**(a+1)\
                                   -2*(j-i)**(a+1))
                elif i==j:#diagonal
                    T[i,j] = coeff
        T = np.transpose(T)
    # methode 3 : puissance de S
    elif method1=='fracpower':
        # Matrice Operateur
        r = a%1
        m = int(a-r)
        T = np.eye(nx)
        S = 1/nx*np.tri(nx)
        Sinv = np.linalg.inv(S)
        for _ in range(m):
            T = T.dot(S)
        if r==0:
            pass
        else:
            Tr = Adaptative_Quad_DE(S,Sinv,r)
            T  = T.dot(Tr)
    # ==================================================
    # Matrice symétrique
    tTT = np.transpose(T).dot(T)
    # ==================================================
    # Matrice regularisation
    if method2=='inv':
        tDD = power(self.tTT,-p/a)
    if method2=='explicit':#default
        if a<=1:
            B        = 2*nx**2*np.diag(np.ones(nx))\
                      -1*nx**2*np.diag(np.ones(nx-1),-1)\
                      -1*nx**2*np.diag(np.ones(nx-1),1)
            B[0,0]   = nx**2
            #B[0,1]   = -2*nx**2
        elif a<=2:
            B        = 6*nx**4*np.diag(np.ones(nx))\
                       -4*nx**4*np.diag(np.ones(nx-1),-1)\
                       -4*nx**4*np.diag(np.ones(nx-1),1)\
                       +1*nx**4*np.diag(np.ones(nx-2),-2)\
                       +1*nx**4*np.diag(np.ones(nx-2),2)
            B[0,0]   = 2*nx**4
            B[0,1]   = -2*nx**4
            B[1,0]   = -2*nx**4
            B[1,1]   = 5*nx**4
            B[-1,-1] = 7*nx**4
            B        = Power(B,1/2)
        else:
            print("Pas implémenté.")
            B = np.eye(nx)
        D   = Power(B,p/2)
        tDD = np.transpose(D).dot(D)
    # ==================================================
    #
    return T,tTT,tDD

def Conjugate_grad(A,b,x=None,eps=1e-16):
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
    if test < 0.001 : # M is symmetric
        D,P = eigh(M)
        D = np.diag(D**r)
        D = P.dot(D).dot(np.transpose(P))
    else: # M is not symmetric, but it has to be invertible
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print('The matrix {} is not invertible'.format(M))
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