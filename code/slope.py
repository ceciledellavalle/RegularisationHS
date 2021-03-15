import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.linalg import inv,pinvh,eig,eigh
from scipy.stats import linregress
from scipy.special import gamma


def Slope(a=1,p=1,nx=100,npt=20):
    np.random.seed(101)
    # power subdunction
    def power(M,r) :
        D,P = eigh(M)
        D   = np.diag(D**r)
        return P.dot(D).dot(np.transpose(P))
    # initialisation
    dx   = 1/nx
    # Matrice opérateur
    T = np.zeros((nx,nx))
    coeff = 1/gamma(a)*1/a*1/(a+1)*1/nx**a
    for i in range(nx):
        for j in range(nx):#lower half
            if i<j:
                T[i,j] = coeff*((j-i+1)**(a+1)\
                               +(j-i-1)**(a+1)\
                               -2*(j-i)**(a+1))
            elif i==j:#diagonal
                T[i,j] = coeff
    T = np.transpose(T)
    tTT = np.transpose(T).dot(T)
    # Matrice regularisation
    D   = power(tTT,-p/(2*a))
    tDD = np.transpose(D).dot(D)
    q   = 2*p+a
    R   = power(tTT,-q/(2*a))
    print("commute coeff : ",np.linalg.norm(tTT.dot(tDD)-tDD.dot(tTT)))
    # Synthetic Data
    t  = np.linspace(0,1-1/nx,nx)
    x  = np.exp(-(t-0.5)**2/0.1**2)
    # x  = x/np.linalg.norm(x)
    rho= np.linalg.norm(R.dot(x))
    y  = T.dot(x)
    # eps
    eps   = np.logspace(-3,-1,npt)
    delta = np.zeros(npt)
    err   = np.zeros(npt)
    # initialise error vector 
    no = np.random.randn(nx)
    no = no*np.linalg.norm(y)/np.linalg.norm(no)
    for i,l in enumerate(eps):
        # step0 : initialisation
        error_compare = 1000
        # step1 : noisy data
        yd = y + l*no
        delta[i] = np.linalg.norm(yd-y)
        # step 2 : optimal alpha
        alpha_op = (delta[i]/rho)**(2*(a+p)/(a+q))
        reg_inf = alpha_op/10
        reg_sup = alpha_op*10
        for alpha in np.linspace(reg_inf,reg_sup,1000*npt):
            # step 3 : inversion
            xd    = np.linalg.inv(tTT + alpha*tDD).dot(np.transpose(T).dot(yd))
            # step 4 : error
            error = np.linalg.norm(xd-x)
            if error < error_compare:
                error_compare = error
                err[i]        = error
                reg           = alpha
        if (reg==reg_inf) or (reg==reg_sup):
            print("Wrong regularization parameter")
            sys.exit()
        if i%5==0:
            plt.figure(figsize=(7, 4))
            plt.subplot(121)
            plt.plot(t,y)
            plt.plot(t,yd)
            plt.subplot(122)
            plt.plot(t,x)
            plt.plot(t,np.linalg.inv(tTT + reg*tDD).dot(np.transpose(T).dot(yd)))
            plt.show()
    # plot
    s,r,R,_,_ = linregress(np.log(delta), np.log(err))
    plt.loglog(delta,err,'r+',label='error')
    plt.loglog(delta,np.exp(r)*delta**s,label="%.3f"%s)
    plt.legend()
    # stat
    q = 2*p+a
    print("th. smax =", q/(a+q),", s = %.2f"%(s), ", R = %.5f"%(R))
    print("th. qmax = ",q ,", q = %.2f"%(s*a/(1-s)))
	#