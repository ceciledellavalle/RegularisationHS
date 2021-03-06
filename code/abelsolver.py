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
from code.myfunc import Export

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
                 folder ='./plots/data'):
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
    def Slope(self,npt=20,interval=(-4,-1),style='gauss',\
              reg_param = (-16,-6),\
              export=False,seed=True):
        """
          Compute the slope of convergence of the error according to the noise level (or standard deviation of noise)
          of the corresponding inverse problem, ill-posed of order a and regularized with order p
             Parameters
             ----------
                   npt          (int): number of point to compute slope
                   interval   (tuple): contains the two boudaries of the noise interval (in log scale)
                   style     (string): define the style of the curve x, 
                                       if 'gauss' then it is a centered gaussian function
                                       if 'offgauss' then it is gaussian like but not centered
                   reg_param  (tuple): contains the two boundaries of the regularization parameter (in log scale)
                   export      (bool): if True, then export curve in self.folder
                   seed        (bool): if True, the random vector is seed, else it is not
            Retruns
            ----------
                    --
        """
        #
        # random initialization
        if seed:
            np.random.seed(101)
        # generate data
        x,y,_         = self.DataGen(export=export,style=style)
        # norm a priori
        q             = 2*self.p+self.a
        R             = Power(self.tDD,q/(2*self.p))
        rho           = np.linalg.norm(R.dot(x))
        # eps
        eps_1,eps_2   = interval
        eps           = np.logspace(eps_1,eps_2,npt)
        delta         = np.zeros(npt)
        err           = np.zeros(npt)
        # initialise error vector 
        no            = np.random.randn(self.nx)
        no            = no*np.linalg.norm(y)/np.linalg.norm(no)
        for i,l in enumerate(eps):
            # step1 : noisy data
            yd        = y + l*no
            delta[i]  = np.linalg.norm(yd-y)
            # step 2 : solve
            reg1,reg2 = reg_param
            reg_inf   = 10**reg1
            reg_sup   = 10**reg2
            if i%6==0:
                xadp = self.Solver(x,yd,reg_inf=reg_inf,reg_sup=reg_sup,verbose=True,delta=delta[i])
                print("==========================================")
            else:
                xadp = self.Solver(x,yd,reg_inf=reg_inf,reg_sup=reg_sup,verbose=False,delta=delta[i])
            # step 3 : compute error
            err[i] = np.linalg.norm(xadp-x)
        # plot
        s,r,Re,_,_ = linregress(np.log(delta[:17]), np.log(err[:17]))
        plt.loglog(delta,err,'r+',label='error')
        plt.loglog(delta,np.exp(r)*delta**s,label="%.3f"%s)
        plt.legend()
        # stat
        print("Statistic for a centered Gaussian : ")
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
    def DataGen(self,noise=0.05,style='gauss',export=False):
        """
          Compute the slope of convergence of the error according to the noise level (or standard deviation of noise)
          of the corresponding inverse problem, ill-posed of order a and regularized with order p
             Parameters
             ----------
                   noise       (float): noise level (in %)
                   style      (string): define the style of the curve x, 
                                        if 'gauss' then it is a centered gaussian function
                                        if 'offgauss' then it is gaussian like but not centered
                   export       (bool): if True, then export curve in self.folder
            Retruns
            ----------
                 (numpy.array): gaussian vector of size nx
                 (numpy.array): image by the Abel operator of a gaussian vector, size nx
                 (numpy.array): noisy image by the Abel operator of a gaussian vector, size nx
        """
        # Synthetic Data
        t    = np.linspace(0,1-1/self.nx,self.nx)
        # gauss
        if style=='gauss':
            x    = np.exp(-(t-0.5)**2/0.1**2)
        elif style=='offgauss':
            x    = np.exp(-(t-0.2)**2/0.2**2)
        else:
            x = np.ones(self.nx)
        #
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
            Export(t,x,self.folder,style+"{}".format(self.a)+kern)
            Export(t,y,self.folder,"T"+style+"{}".format(self.a)+kern)
            Export(t,yd,self.folder,"T"+style+"n"+"{}".format(self.a)+kern)
        return x,y,yd


    # Solve for one couple (x,y,rho)    
    def Solver(self,x,y,reg_inf=10**-8,reg_sup=10**-3,delta=0,\
               verbose=True,warning=False,export=False):
        """
          Solve the inverse problem for a vector x, and image y that may or may not be noisy.
          
             Parameters
             ----------
                   x   (numpy.array): exact function to reconstruct, size nx
                   y   (numpy.array): image of x or noisy image of x, size nx
                   reg_inf   (float): lower boundary for regularization parameter
                   reg_sup   (float): higher boundary for regularization parameter
                   verbose    (bool): if True, print statistic and plot reconstructed function
                   warning    (bool): if True, print warning if the regularization parameter saturates
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
        # step 5 : alert and statistics
        # warning
        if warning:
            if (reg==reg_inf): 
                print("noise={:.3e}".format(delta),\
                      "inf=",reg_inf,", reg=",reg)
                print("Wrong regularization parameter, too high")
                print("==========================================")
            elif (reg==reg_sup):
                print("noise={:.3e}".format(delta),\
                      "sup=",reg_sup,", reg=",reg)
                print("Wrong regularization parameter, too low")
                print("==========================================")
        # verbose and plots
        if verbose:
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
            Export(t,xadp,self.folder,"pred{}{}".format(self.a,self.p)+kern)
        # step 6 : return
        return xadp
    
    # Solve for one couple (x,y,rho)    
    def BlindSolver(self,y,reg_inf=10**-6,reg_sup=10**-3,\
               verbose=True,warning=False,export=False):
        """
          Solve the inverse problem for image y that may or may not be noisy,
          and the regularisation paramter alpha is blindly found.
          
             Parameters
             ----------
                   y   (numpy.array): image of x or noisy image of x, size nx
                   reg_inf   (float): lower boundary for regularization parameter
                   reg_sup   (float): higher boundary for regularization parameter
                   verbose    (bool): if True, print statistic and plot reconstructed function
                   warning    (bool): if True, print warning if the regularization parameter saturates
                   export     (bool): export datas
            Retruns
            ----------
                 (numpy.array): solution of the regularized inverse problem, size nx
        """
        # step 1 : estimating delta
        delta = self.nx*np.linalg.norm(y[:20])/20       
        reg   = 0
        error_compare = delta
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
            Txd   = self.T.dot(xd)
            error = np.linalg.norm(Txd-y)
            if error < error_compare:
                error_compare = error
                reg           = alpha
                xadp          = xd.copy()
        # step 5 : alert and statistics
        # warning
        if warning:
            if (reg==reg_inf): 
                print("noise={:.3e}".format(delta),\
                      "inf=",reg_inf,", reg=",reg)
                print("Wrong regularization parameter, too high")
                print("==========================================")
            elif (reg==reg_sup):
                print("noise={:.3e}".format(delta),\
                      "sup=",reg_sup,", reg=",reg)
                print("Wrong regularization parameter, too low")
                print("==========================================")
        # verbose and plots
        if verbose:
            print("delta={:.3e}, inf={:.3e}, sup={:.3e}, reg={:.3e}".format(\
                     delta,reg_inf,reg_sup,reg))
            t = np.linspace(0,1,self.nx)
            plt.figure(figsize=(7, 4))
            plt.subplot(121)
            plt.plot(t,y)
            plt.subplot(122)
            plt.plot(t,xadp)
            plt.show()
        # export
        if export:
            if self.kern==True:
                kern = 'kernel'
            else:
                kern=''
            Export(t,xadp,self.folder,"pred{}{}".format(self.a,self.p)+kern)
        # step 6 : return
        return xadp, delta
    
