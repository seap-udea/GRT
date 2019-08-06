#################################################################################
#     ______                 ____             ___                               # 
#    / ____/________ __   __/ __ \____ ___  _|__ \                              #
#   / / __/ ___/ __ `/ | / / /_/ / __ `/ / / /_/ /                              #
#  / /_/ / /  / /_/ /| |/ / _, _/ /_/ / /_/ / __/                               #
#  \____/_/   \__,_/ |___/_/ |_|\__,_/\__, /____/                               #
#                                    /____/                                     #
#################################################################################
# Jorge I. Zuluaga (C) 2019                                                     #
#################################################################################
#!/usr/bin/env python
# coding: utf-8

# # GravRay Test Statistical Module

from scipy.stats import multivariate_normal as multinorm
from gravray import *
from gravray.util import *

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

class MultiVariate(object):
    """
    A multivariate distribution combining multivariate gaussian distributed variables and 
    uniformly distributed (independent) variables.
    
    Variables does not need to be ordered.  This is the aim of the "description" attribute:
    to indicate which variables follow the multinormal distribution and which are uniformly
    distributed.
    
    Initialization attributes:
        description: describe type of the variable (eg. [1,1,0] first two are normally distributed
        the last one is uniformly distributed), list (n)

    Other attributes:
        n: number of variables of the distribution (len(description)), int
        G: number of normally distributed variables, int
        U: number of uniformly distributed variables, int
        inormal: indexes of the normally distributed variables, numpy array (G)
        iuniform: indexes of the uniformly distributed variables, numpy array (U)
        
    Derivative attributes:
        M: number of normal distributions used for the normal variables, int 
        N: number of parameters of the normal part of the distribution N = 10M-1, int
        locs: means of the normal variables, numpy array (MxG)
        scales: std. of the normal variables, numpy array (MxG)
        angles: angles of orientation of the normal variables, numpy array (MxG), radians
        covs: covariance matrices, list of numpy arrays (MxGxG)
        ranges: ranges of the uniformly distributed variables, list of numpy arrays (Ux2)
        uvalues: value of the probability of the uniform variables, numpy array (U)
    
    Methods:
        setUnflatten: set parameters of the distribution from unflatten set of parameters.
        setFlatten: set parameters of the distribution from flatten set of parameters.
        pdf: compute the value of the probability.
        rvs: generate samples.
        
    Examples:
        P=MultiVariate([1,1,1,0,0])
        weights=[0.6]
        locs=[
            [0.5,0.5,-2.0],
            [2.0,0.3,-2.6]
        ]
        scales=[
            [1.3,0.7,0.5],
            [0.4,0.9,1.6]
        ]
        angles=[
            [-40.0*Angle.Deg,-86.0*Angle.Deg,0.0*Angle.Deg],
            [+80.0*Angle.Deg,-109.0*Angle.Deg,0.0*Angle.Deg]
        ]
        ranges=[
            [-1.0,1.0],
            [0.0,2*np.pi]
        ]
        P.setUnflatten(weights,locs,scales,angles,ranges)
        P.pdf(np.array([0,0,0,0,0]))
        P.rvs(1000)
        
        params=[
            #weights
            0.6,
            #locs
            0.5, 0.5, -2.0,
            2.0, 0.3, -2.6, 
            #scales
            1.3, 0.7, 0.5,
            0.4, 0.9, 1.6, 
            #Angles
            -40.0*Angle.Deg,-86.0*Angle.Deg,0.0*Angle.Deg,
            +80.0*Angle.Deg,-109.0*Angle.Deg,0.0*Angle.Deg,
            #Ranges
            -1.0,1.0,
            0.0,2*np.pi
        ]
        P.setFlatten(params)
        P.pdf(np.array([0,0,0,0,0]))
        P.rvs(1000)        
    """
    #Constants
    Maxls=10.0

    def __init__(self,description):
        
        #Basic attributes
        self.description=np.array(description)
        self.n=len(self.description)
        indexes=np.arange(self.n)
        self.inormal=indexes[self.description==1]
        self.G=len(self.inormal)
        self.iuniform=indexes[self.description==0]
        self.U=len(self.iuniform)
        
        #Key attributes are set to None awaiting for the set method
        self.params=None
        self.locs=None
        self.covs=None
        self.aweights=None
        self.uvalues=None
        
    def pdf(self,r):
        """
        Compute the PDF.
        
        Parameter:
            r: point in the n-dimensional space, numpy array (n)
        
        Return:
            p: pdf.
        """
        rnormal=r[self.inormal]    
        runiform=r[self.iuniform]
        value=0
        try:
            for w,loc,cov in zip(self.aweights,self.locs,self.covs):
                value+=w*multinorm.pdf(r[self.inormal],loc,cov)
            value*=self.uvalues.prod()
            return value
        except Exception as e:
            errorMsg(e,"You must first set the parameters, eg. MultiVariate.setUnflatten")
            raise
            
    def rvs(self,N):
        """
        Generate a random sample of points following this Multivariate distribution.
        
        Parameter:
            N: number of samples.
            
        Return:
            rs: samples, numpy array (Nxn)
        """
        rs=np.zeros((N,self.n))
        try:
            n=len(self.covs)
        except Exception as e:
            errorMsg(e,"You must first set the parameters, eg. MultiVariate.setUnflatten")
            raise
        for i in range(N):
            n=Util.genIndex(self.aweights)
            rs[i,self.inormal]=multinorm.rvs(self.locs[n],self.covs[n])
            rs[i,self.iuniform]=np.array([np.random.uniform(low=self.ranges[i][0],high=self.ranges[i][1])                                   for i in range(self.U)])
        return rs

    def setUnflatten(self,weights,locs,scales,angles,ranges):
        """
        Set the parameters of the distribution individually.
        
        Parameters:
            weights: weights of the gaussian distributions, list (M-1)
                Ex. [0.6,0.2], when using three gaussians with weights 0.6, 0.2 (and 0.1)
            locs: means, list (M).
                Ex. [[0.1,1,2],[-2,1,0],[2,3,1]], 3 gaussians of 3 variables.
            scales: standard deviations, list of lists (M x G), all positive.
                Ex. [[0.1,0.4,0.5],[0.5,0.9,0.2],[0.1,0.3,0.3]], 3 gaussians of 3 variables.
            angles: euler angles of the distribution, list of lists (M x G), radians.
                Ex. [[0.1,0.4,0.5],[0.5,0.9,0.2],[0.1,0.3,0.3]], 3 gaussians of 3 variables.
            ranges: ranges of the uniformly distributed variables, list of lists (U x 2)
                Ex. [[-1.0,+1.0],[0.0,2*np.pi]], 2 uniform variables.
            
        Return: None.
        """
        #Store 
        self.M=len(locs)
        self.weights=np.array(weights)
        if self.M==1:
            self.aweights=np.array([1.0])
        else:
            self.aweights=np.concatenate((self.weights,[1-sum(self.weights)]))
        self.locs=np.array(locs)
        self.scales=np.array(scales)
        self.angles=np.array(angles)
        
        #Check consistency with initialization
        if self.G!=len(locs[0]):
            raise AttributeError(f"Locations provided ({len(locs[0])}) is not compatible with dimensions ({self.G})")
        
        #Covariances
        self._calcCovariances()

        #Constraints for minimization
        self.calcBounds(self.Maxls)

        #Uniform deviated variables
        self.ranges=ranges
        if self.U!=len(ranges):
            raise AttributeError(f"Ranges provided ({len(ranges)}) is not compatible with dimension ({self.U})")
        self.uvalues=np.array([1/(self.ranges[i][1]-self.ranges[i][0]) for i in range(self.U)])

        #Full set of parameters
        self.params=sum([weights],[])+sum(locs,[])+sum(scales,[])+sum(angles,[])+sum(ranges,[])
        self.N=len(self.params)-2*self.U
        
    def setFlatten(self,params):
        """
        Set the parameters of the distribution individually.
        
        Parameters:
            params: all params in one list, list (10M-1+2U)
         
        Return: None.
        """

        #Unflatten
        self.params=np.array(params)
        
        #Get ranges
        self.ranges=self.params[-2*self.U:].reshape(self.U,2)
        self.uvalues=np.array([1/(self.ranges[i][1]-self.ranges[i][0]) for i in range(self.U)])
        
        #Get gaussian parameters
        N=len(self.params)-2*self.U
        if N>9:
            M=np.int((len(params)+1)/10)
            i=0;j=i+M-1
            weights=list(params[i:j])+[1-np.sum(params[i:j])]
            self.weights=np.array(weights)
            self.aweights=self.weights+[1-sum(self.weights)]
        else:
            #Case for one function
            M=1
            self.weights=np.array([1.0])
            self.aweights=np.array([1.0])            
            j=0
        i=j;j=i+3*M
        self.locs=np.reshape(params[i:j],(M,3))
        i=j;j=i+3*M
        self.scales=np.reshape(params[i:j],(M,3))
        i=j;j=i+3*M
        self.angles=np.reshape(params[i:j],(M,3))

        self.N=N
        self.M=M
        
        #Covariances
        self._calcCovariances()

        #Constraints for minimization
        self.calcBounds(self.Maxls)

    def calcBounds(self,maxls):
        """
        Compute the boundaries of the parameters.
        
        Parameters:
            maxls: maximum (and minimum=-maximum) value of the locs and the stds.
         
        Return: None.
        """
        
        M=self.M
        wbnds=(0,1),
        lbnds=(-maxls,maxls),
        sbnds=(1e-3,maxls),
        abnds=(-np.pi,np.pi),
        self.bounds=()
        if M>1:
            self.bounds=wbnds*(M-1)
        self.bounds+=lbnds*M*3+sbnds*M*3+abnds*M*3
        
    def _calcCovariances(self):
        """
        Compute covariance matrices from the stds and the angles.
        
        Sources: https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/    
        """
        rots=[]
        self.covs=[]
        for scale,angle in zip(self.scales,self.angles):
            L=np.identity(len(scale))*np.outer(np.ones(len(scale)),scale)
            spy.eul2m(-angle[0],-angle[1],-angle[2],3,1,3)
            rots+=[spy.eul2m(-angle[0],-angle[1],-angle[2],3,1,3)]
            self.covs+=[spy.mxm(spy.mxm(rots[-1],spy.mxm(L,L)),spy.invert(rots[-1]))]

