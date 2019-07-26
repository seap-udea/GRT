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

from scipy.stats import multivariate_normal as multinorm
from gravray import *
from gravray.util import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

class MultiNormal(object):
    
    #Maximum range of locations and scales for bounds (intended for contraining minimization)
    MAXLS=10.0

    def pdf(self,r):
        value=0
        for w,loc,cov in zip(self.aweights,self.locs,self.covs):
            value+=w*multinorm.pdf(r,loc,cov)
        return value

    def rvs(self,N):
        """
        Generate a random sample of points following this MND
        """
        rs=[]
        for i in range(N):
            n=Util.genIndex(self.aweights)
            r=multinorm.rvs(self.locs[n],self.covs[n])
            rs+=[r]
        return rs

    def setUnflatten(self,weights,locs,scales,angles):

        #Store 
        self.weights=weights
        self.aweights=self.weights+[1-sum(self.weights)]
        self.locs=locs
        self.scales=scales
        self.angles=angles
        self.M=len(locs)
        self.params=sum([weights],[])+sum(locs,[])+sum(scales,[])+sum(angles,[])
        self.N=len(self.params)
        
        #Covariances
        self._calcCovariances()
        
        #Constraints for minimization
        self.calcBounds(self.MAXLS)
            
    def setFlatten(self,params):

        #Unflatten
        self.params=params
        N=len(self.params)
        if N>9:
            M=np.int((len(params)+1)/10)
            i=0;j=i+M-1
            weights=list(params[i:j])+[1-np.sum(params[i:j])]
        else:
            #Case for one function
            M=1
            weights=[1.0]
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
        self.calcBounds(self.MAXLS)

    def calcBounds(self,maxls):
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
        rots=[]
        self.covs=[]
        for scale,angle in zip(self.scales,self.angles):
            L=np.identity(len(scale))*np.outer(np.ones(len(scale)),scale)
            spy.eul2m(-angle[0],-angle[1],-angle[2],3,1,3)
            rots+=[spy.eul2m(-angle[0],-angle[1],-angle[2],3,1,3)]
            self.covs+=[spy.mxm(spy.mxm(rots[-1],spy.mxm(L,L)),spy.invert(rots[-1]))]

