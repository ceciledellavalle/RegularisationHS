import numpy as np
import math

def GetInterval(A,Ainv,beta,eps=10**-15):
    normA  = np.linalg.norm(A)
    normAi = np.linalg.norm(Ainv)
    # compute a
    c  = beta*math.pi
    a1 = (c*(1+beta)*eps)/(4*math.sin(c)*(1+2*beta))
    a2 = (2*normAi)**-beta
    a  = min(a1,a2)
    # compute b
    d  = beta/(beta-1)
    e  = (1-beta)*math.pi
    b1 = (e*(2-beta)*eps)**d/(4*math.sin(c)*(3-2*beta)*normA)**d
    b2 = (2*normA)**beta
    b  = max(b1,b2)
    # compute interval [l,r]
    l = math.asinh(2*math.log(a)/c)
    r = math.asinh(2*math.log(b)/c)
    return l, r

def FDE(x,A,beta):
    nx     = A.shape[0]
    try:
        f0  = math.exp((beta-1)*math.pi*math.sinh(x)/2)
    except OverflowError:
        f0  = float('inf')
    try:
        f1  = math.exp(math.pi*math.sinh(x)/2)
    except OverflowError:
        f1  = float('inf')
    try:
        f2 = math.cosh(x)
    except OverflowError:
        f2 = float('inf')
    M      = f0*f2*np.linalg.inv(np.eye(nx)+1/f1*A)
    return M

def Adaptative_Quad_DE(A,Ainv,beta,eps=10**-5,niter=5):
    l,r = GetInterval(A,Ainv,beta,eps)
    m   = 8
    h   = (r-l)/(m-1)
    g   = math.sin(beta*math.pi)/2
    # compute T
    T   = h*(FDE(l,A,beta)+FDE(r,A,beta))/2
    for k in range(1,m-1):
        T += h*FDE(l+k*h,A,beta)
    Told = T.copy() 
    # compute quadrature
    for s in range(niter):
        h = h/2
        T = T/2
        for k in range(1,m-1):
            T += h*FDE(l+(2*k-1)*h,A,beta)
        m = 2*m-1
        # test if precision is obtained
        err = g*np.linalg.norm(A.dot(T)-A.dot(Told))
        if abs(err)<eps/2:
            return g*A.dot(T)
        Told = T.copy()
    # return
    return g*A.dot(T)  