"""
AbelSolver model classes.
Classes
-------
    AbelSolver : Tikhonov solver for Abel operator
@author: Cecile Della Valle
@date: 03/01/2021
"""

#global import
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.linalg import inv,pinvh,eig,eigh
from scipy.stats import linregress
from scipy.special import gamma
from scipy.sparse.linalg import cg
#local import
from code.myfunc import Conjugate_grad
from code.myfunc import Power
from code.myfunc import MatrixGen

# Solver
class AbelSolver():
    """
        Includes the main training and testing methods of iRestNet.
        Attributes
        ----------
            a          (float): order of ill-posedness
            p          (float): order of regularization
            nx           (int): discretization time step
            kern        (bool): True if the Abel operator includes a kernel k(t,s)= 1/(t+s)^0.5, 
                                False otherwise, and k(t,s)=1
            methT        (str): 'trapeze' for the trapeze approximation of T, 
                                 is also implemented the finite element method ('eltP0') 
                                 and the computation using semi-groups properties ('fracpower')
            methB        (str): 'explicit' for the finite difference method to approximate B,
                                 but can also be define as (tTT)^-1 ('inv')
            resol        (str): specify the resolution method, either the matrix inversion with numpy.linalg.inv method,
                                or the scipy conjugate grandient method ('cg'), or my own implementation of the conjugate gradient ('mycg')
            folder       (str): folder to export datas and plots
            T    (numpy.array): Abel operator size nx,nx
            tTT  (numpy.array): symmetric of Abel operator size nx,nx
            tDD  (numpy.array): regularisation operator size nx,nx
        """
    # initialization of one inverse problem
    def __init__(self,a=1,p=1,nx=100,kernel=False,npt=20,\
                 methT='trapeze',methB='explicit',resol='no',\
                 folder ='./../redaction/data'):
        """
            Define one instance of the inverse problem of order a 
                 with regularization by the generalized Tikhonov method of order p.
            Parameters
            ----------
                a          (float): order of ill-posedness
                p          (float): order of regularization
                nx           (int): discretization time step
                kernel      (bool): True if the Abel operator includes a kernel k(t,s)= 1/(t+s)^0.5, 
                                    False otherwise, and k(t,s)=1
                methT        (str): 'trapeze' for the trapeze approximation of T, 
                                     is also implemented the finite element method ('eltP0') 
                                     and the computation using semi-groups properties ('fracpower')
                methB        (str): 'explicit' for the finite difference method to approximate B,
                                     but can also be define as (tTT)^-1 ('inv')
                resol        (str): specify the resolution method, either the matrix inversion with numpy.linalg.inv method,
                                    or the scipy conjugate grandient method ('cg'), or my own implementation of the conjugate gradient ('mycg')
                folder       (str): folder to export datas and plots
        """
        # random initialization
        np.random.seed(32)
        # physics
        self.a = a
        self.p = p
        self.kern = kernel
        # numerics
        self.nx = nx
        # resolution parameters
        self.methT = methT
        self.methB = methB
        self.resol = resol
        # export parameters
        self.folder = folder
        # operators
        self.T, self.tTT, self.tDD = MatrixGen(a,p,nx,kernel,methT,methB)
        
    # Compute and export slope
    def Slope(self,npt=20,export=False):
        """
          Compute the slope of convergence of the error according to the noise level (or standard deviation of noise)
          of the corresponding inverse problem, ill-posed of order a and regularized with order p
             Parameters
             ----------
                   npt           (int): number of point to compute slope
                   folder       (str): folder to export datas and plots
            Retruns
            ----------
                    --
        """
        #
        x,y,_         = self.DataGen(export=export)
        # norm a priori
        q             = 2*self.p+self.a
        R             = Power(self.tDD,q/(2*self.p))
        rho           = np.linalg.norm(R.dot(x))
        # eps
        eps           = np.logspace(-3,-1,npt)
        delta         = np.zeros(npt)
        err           = np.zeros(npt)
        # initialise error vector 
        no            = np.random.randn(self.nx)
        no            = no*np.linalg.norm(y)/np.linalg.norm(no)
        for i,l in enumerate(eps):
            # step1 : noisy data
            yd            = y + l*no
            delta[i]      = np.linalg.norm(yd-y)
            # step 2 : optimal alpha
            #alpha_op      = (delta[i]/rho)**(2*(a+p)/(a+q))
            reg_inf       = 10**-16 #alpha_op/10
            reg_sup       = 10**-6 #alpha_op/10
            #
            if i%10==0:
                xadp = self.Solver(x,yd,reg_inf=reg_inf,reg_sup=reg_sup,verbose=True)
                print("==========================================")
            else:
                xadp = self.Solver(x,yd,reg_inf=reg_inf,reg_sup=reg_sup,verbose=False)
            # step 3 : compute error
            err[i] = np.linalg.norm(xadp-x)
        # plot
        s,r,Re,_,_ = linregress(np.log(delta[:15]), np.log(err[:15]))
        plt.loglog(delta,err,'r+',label='error')
        plt.loglog(delta,np.exp(r)*delta**s,label="%.3f"%s)
        plt.legend()
        # stat
        print("th. smax =", q/(self.a+q),", s = %.2f"%(s), ", R = %.5f"%(Re))
        print("th. qmax = ",q ,", q = %.2f"%(s*self.a/(1-s)))
        # export
        if export:
            if self.kern==True:
                kern = 'kernel'
            else:
                kern=''
            Export(delta,err,self.folder,"error_a{}p{}".format(self.a,self.p)+kern)
            Export(delta,np.exp(r)*delta**s,self.folder,"errorline_a{}p{}".format(self.a,self.p)+kern)

    # Generate vector x and y
    def DataGen(self,noise=0.05,export=False):
        """
          Compute the slope of convergence of the error according to the noise level (or standard deviation of noise)
          of the corresponding inverse problem, ill-posed of order a and regularized with order p
             Parameters
             ----------
                   npt           (int): number of point to compute slope
                   folder       (str): folder to export datas and plots
            Retruns
            ----------
                 (numpy.array): gaussian vector of size nx
                 (numpy.array): image by the Abel operator of a gaussian vector, size nx
                 (numpy.array): noisy image by the Abel operator of a gaussian vector, size nx
        """
        np.random.seed(32)
        # Synthetic Data
        t    = np.linspace(0,1-1/self.nx,self.nx)
        x    = np.exp(-(t-0.5)**2/0.1**2)
        x    = x/np.amax(x)
        y    = self.T.dot(x)
        # add noise
        no   = np.random.randn(self.nx)
        no   = no*np.linalg.norm(y)/np.linalg.norm(no)
        yd   = y + noise*no
        # export
        if export:
            if self.kern==True:
                kern = 'kernel'
            else:
                kern=''
            Export(t,x,self.folder,"gauss{}".format(a)+kern)
            Export(t,y,self.folder,"Tgauss{}".format(a)+kern)
            Export(t,yd,self.folder,"Tgaussn{}".format(a)+kern)
        return x,y,yd


    # Solve for one couple (x,y,rho)    
    def Solver(self,x,y,reg_inf=10**-8,reg_sup=10**-3,verbose=True,export=False):
        """
          Solve the inverse problem for a vector x, and image y that may or may not be noisy.
          
             Parameters
             ----------
                   x   (numpy.array): exact function to reconstruct, size nx
                   y   (numpy.array): image of x or noisy image of x, size nx
                   reg_inf   (float): lower boundary for regularization parameter
                   reg_sup   (float): higher boundary for regularization parameter
                   verbose    (bool): print statistic and plot reconstructed function
                   export     (bool): export datas
            Retruns
            ----------
                 (numpy.array): solution of the regularized inverse problem, size nx
        """
        # step 1 : initialisation
        error_compare = 1000
        reg           = 0
        # step 2 :loop over alpha to find the best candidate of regularization
        for alpha in np.linspace(reg_inf,reg_sup,10000):
            # step 3 : inversion
            if self.resol == 'cg':
                xd = cg(self.tTT + alpha*self.tDD,np.transpose(self.T).dot(y))
            if self.resol == 'mycg':
                xd,_ = Conjugate_grad(self.tTT + alpha*self.tDD,np.transpose(self.T).dot(y))
            else:
                xd = np.linalg.inv(self.tTT + alpha*self.tDD).dot(np.transpose(self.T).dot(y))
            # step 4 : error computation
            error = np.linalg.norm(xd-x)
            if error < error_compare:
                error_compare = error
                reg           = alpha
                xadp          = xd.copy()
        # warning
        if (reg==reg_inf): 
            print("inf=",reg_inf,", reg=",reg)
            print("Wrong regularization parameter, too high")
            print("==========================================")
        if (reg==reg_sup):
            print("sup=",reg_sup,", reg=",reg)
            print("Wrong regularization parameter, too low")
            print("==========================================")
        # verbose and plots
        if verbose==True:
            print("err={:.3e}, inf={:.3e}, sup={:.3e}, reg={:.3e}".format(\
                     error_compare,reg_inf,reg_sup,reg))
            t = np.linspace(0,1,self.nx)
            plt.figure(figsize=(7, 4))
            plt.subplot(121)
            plt.plot(t,y)
            plt.subplot(122)
            plt.plot(t,x)
            plt.plot(t,xadp)
            plt.show()
        # export
        if export:
            if self.kern==True:
                kern = 'kernel'
            else:
                kern=''
            Export(t,xadp,"gauss_pred{}{}".format(self.a,self.p)+kern)
        # step 6 : return
        return xadp
    

