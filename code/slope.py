import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.linalg import inv,pinvh,eig,eigh
from scipy.stats import linregress
from scipy.special import gamma
from code.fracpower import Adaptative_Quad_DE

# Compute and export slope
def Slope(a=1,p=1,nx=100,npt=20,methT='trapeze',methB='inv',export=False):
    #
    np.random.seed(32)
    # power subdunction
    def power(M,r) :
        D,P = eigh(M)
        D = np.diag(D**r)
        return P.dot(D).dot(np.transpose(P))
    #
    T,tTT,tDD     = MatrixGen(a=a,p=p,nx=nx,method1=methT,method2=methB)
    x,y,_         = DataGen(T)
    # norm a priori
    q             = 2*p+a
    R             = power(tDD,q/(2*p))
    rho           = np.linalg.norm(R.dot(x))
    # eps
    eps           = np.logspace(-4,-1,npt)
    delta         = np.zeros(npt)
    err           = np.zeros(npt)
    # initialise error vector 
    no            = np.random.randn(nx)
    no            = no*np.linalg.norm(y)/np.linalg.norm(no)
    for i,l in enumerate(eps):
        # step0 : initialisation
        error_compare = 1000
        # step1 : noisy data
        yd            = y + l*no
        delta[i]      = np.linalg.norm(yd-y)
        # step 2 : optimal alpha
        alpha_op      = (delta[i]/rho)**(2*(a+p)/(a+q))
        reg_inf       = alpha_op/10
        reg_sup       = alpha_op/10
        for alpha in np.linspace(reg_inf,reg_sup,1000*npt):
            # step 3 : inversion
            xd = np.linalg.inv(tTT + alpha*tDD).dot(np.transpose(T).dot(yd))
            # step 4 : error
            error = np.linalg.norm(xd-x)
            if error < error_compare:
                error_compare = error
                err[i]        = error
                reg           = alpha
        if (reg==reg_inf): 
            print("inf=",reg_inf,", reg=",reg)
            print("Wrong regularization parameter, too high")
            print("==========================================")
        if (reg==reg_sup):
            print("sup=",reg_sup,", reg=",reg)
            print("Wrong regularization parameter, too low")
            print("==========================================")
        if i%5==0:
            print("delta={:.3e}, inf={:.3e}, sup={:.3e}, reg={:.3e}".format(\
                         delta[i],reg_inf,reg_sup,reg))
            t = np.linspace(0,1-1/nx,nx)
            plt.figure(figsize=(7, 4))
            plt.subplot(121)
            plt.plot(t,y)
            plt.plot(t,yd)
            plt.subplot(122)
            plt.plot(t,x)
            plt.plot(t,np.linalg.inv(tTT + reg*tDD).dot(np.transpose(T).dot(yd)))
            plt.show()
            print("==========================================")
    # plot
    s,r,Re,_,_ = linregress(np.log(delta[:15]), np.log(err[:15]))
    plt.loglog(delta,err,'r+',label='error')
    plt.loglog(delta,np.exp(r)*delta**s,label="%.3f"%s)
    plt.legend()
    # stat
    print("th. smax =", q/(a+q),", s = %.2f"%(s), ", R = %.5f"%(Re))
    print("th. qmax = ",q ,", q = %.2f"%(s*a/(1-s)))
    # export
    if export:
        Export(delta,err,"error_a{}p{}".format(a,p))
        Export(delta,np.exp(r)*delta**s,"errorline_a{}p{}".format(a,p))
	#

# Generate vector x and y
def DataGen(T,noise=0.05,export=False):
    #
    np.random.seed(101)
    # Synthetic Data
    nx,_ = T.shape
    t    = np.linspace(0,1-1/nx,nx)
    x    = np.exp(-(t-0.5)**2/0.1**2)
    x    = x/np.amax(x)
    y    = T.dot(x)
    # add noise
    no   = np.random.randn(nx)
    no   = no*np.linalg.norm(y)/np.linalg.norm(no)
    yd   = y + noise*no
    # export
    if export:
        Export(t,x,"gauss{}".format(a))
        Export(t,y,"Tgauss{}".format(a))
        Export(t,yd,"Tgaussn{}".format(a))
    return x,y,yd

# Generate the discretized operator T and D
def MatrixGen(a=1,p=1,nx=100,method1='trapeze',method2='inv'):
    # power subdunction
    def power(M,r) :
        D,P = eigh(M)
        D = np.diag(D**r)
        return P.dot(D).dot(np.transpose(P))
    # ==================================================
    # Matrice opérateur
    # methode 1 : trapeze
    if method1=='trapeze':
        T = np.zeros((nx,nx))
        coeff = 1/(2*a)*nx**-a
        for i in range(nx):
            for j in range(nx):#lower half
                if i<j:
                   T[i,j] = coeff*((j-i+1)**(a)\
                               -(j-i-1)**(a))
                elif i==j:#diagonal
                    T[i,j] = coeff
        T = np.transpose(T)
    # methode 2 : element fini P0
    if method1=='eltP0':
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
    if method1=='fracpower':
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
        tDD = power(tTT,-p/a)
    if method2=='explicit':
        if a<1:
            B = 2*nx**2*np.diag(np.ones(nx))\
              -1*nx**2*np.diag(np.ones(nx-1),-1)\
              -1*nx**2*np.diag(np.ones(nx-1),1)
            B[0,0] = nx**2
        elif a<2:
            B = 6*nx**4*np.diag(np.ones(nx))\
              -4*nx**4*np.diag(np.ones(nx-1),-1)\
              -4*nx**4*np.diag(np.ones(nx-1),1)\
              +1*nx**4*np.diag(np.ones(nx-2),-2)\
              +1*nx**4*np.diag(np.ones(nx-2),2)
            B[0,0] = 3*nx**4
            B[0,1] = -2*nx**4
            B[0,2] = -1*nx**4
            B[1,0] = -2*nx**4
            B[1,1] = 3*nx**4
            B[1,2] = -2*nx**4
            B[1,3] = nx**4
        else:
            print("Pas implémenté.")
            B = np.eye(nx)
        tDD = power(B,p)
    # ==================================================
    #
    return T,tTT,tDD

# Solve for one couple (x,y,rho)    
def Solver(x,y,Matrix,reg_inf=10**-10,reg_sup=0.5,export=False):
    #
    a         = 3
    p         = 1
    nx        = x.size
    T,tTT,tDD = Matrix
    #
    # step 1 : initialisation
    error_compare = 1000
    reg           = 0
    #
    for alpha in np.linspace(reg_inf,reg_sup,2000):
        # step 2 : inversion
        xd    = np.linalg.inv(tTT + alpha*tDD).dot(np.transpose(T).dot(y))
        # step 3 : error
        error = np.linalg.norm(xd-x)
        if error < error_compare:
            error_compare = error
            reg           = alpha
    #
    print("err={:.3e}, inf={:.3e}, sup={:.3e}, reg={:.3e}".format(\
                 error_compare,reg_inf,reg_sup,reg))
    xadp = np.linalg.inv(tTT + reg*tDD).dot(np.transpose(T).dot(y))
    # warning
    if (reg==reg_inf): 
        print("inf=",reg_inf,", reg=",reg)
        print("Wrong regularization parameter, too high")
    if (reg==reg_sup):
        print("sup=",reg_sup,", reg=",reg)
        print("Wrong regularization parameter, too low") 
    # plot
    t = np.linspace(0,1,nx)
    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plt.plot(t,y)
    plt.subplot(122)
    plt.plot(t,x)
    plt.plot(t,xadp)
    plt.show()
    # export
    if export:
        Export(t,xadp,"gauss_pred{}{}".format(a,p))
    return x, xadp
    
def Export(x,y,name):
    # initialisation
    folder ='./../redaction/data' #fichier
    Npoint = np.size(x)
    with open(folder+'/'+name+'.txt', 'w') as f:
        f.writelines('xdata ydata \n')
        for i in range(Npoint):
            web_browsers = ['{0}'.format(x[i]),' ','{0} \n'.format(y[i])]
            f.writelines(web_browsers)
