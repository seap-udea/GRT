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

# # GravRay Sampling Module

from gravray import *

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# This module is based on fibpy by Martin Roberts, source code: https://github.com/matt77hias/fibpy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib nbagg')

get_ipython().magic('matplotlib nbagg')

class Sample(object):
    """
    Class of sample
    
    Initialization attributes:
        N: Number of samples, int.
        
    Secondary attributes:
        dim: Dimension of samples, int.
        ss: Samples in cartesian coordinates, numpy array (Nx3)
        pp: Samples in polar or spherical coordinates, numpy array (Nx3)
        dmin, dmed, dmax: Minimum, median and maximum distance, float

    Private attribute:
        _purge: Does the sample need to be purged? (default True), boolean
        
    Other attributes:
        ds: minimum distances of all points, numpy float (N)
        dran: range of distances, float
        dstar: measure of distances (sqrt(N)*dmed), float
            NOTE: Typically this value is between 2.4 and 3.4    
        cargs: arguments for the circumference in polar, dictionary
        wargs: arguments for the sphere in 3d space, dictionary
    """
    def __init__(self,N):
        #Basic
        self.N=N
        
        #Derivative
        self.dim=0
        self.ss=None
        self.pp=None
        self.dmin=self.dmed=self.dmax=0
        
        #Purge
        self._purge=True
        
        #Plotting
        self.cargs=dict(color="k",fill=False,alpha=0.3)
        self.wargs=dict(color="k",lw=0.1)
    
    def _closestDistance(self,r,rs):
        """
        Get the minimum distance from point p to other points
        
        Parameter:
            r: coordinates of the point, numpy array (3)
            rs: coordinates of the points, numpy array (N)
        
        Return:
            dmin: minimum distance, float
        """
        deltas=rs-r
        dist=np.einsum('ij,ij->i',deltas,deltas)
        imin=np.argsort(dist)[1]
        return np.sqrt(dist[imin])

    def _calcDistances(self):
        """
        Calculate the minimum distances of all points in the sample
        """
        self.ds=np.array([self._closestDistance(self.ss[i],self.ss) for i in range(len(self.ss))])
        self.dmin=self.ds.min()
        self.dmax=self.ds.max()
        self.dmed=np.median(self.ds)
        self.dran=self.dmax-self.dmin
        self.dstar=np.sqrt(self.N)*self.dmed
    
    def purgeSample(self,tol=0.5):
        """
        Purge sample, ie. remove points close than a given threshold.
        
        Optional parameters:
            tol: distance to purge, ie. if dmin<tol*dmed then purge, float
            
        Return: None
        """
        while self._purge:
            self._calcDistances()
            if self.dmin<tol*self.dmed:
                ipurge=np.argsort(self.ds)[0]
                self.ss=np.delete(self.ss,ipurge,0)
                self.pp=np.delete(self.pp,ipurge,0)
                self.N-=1
            else:
                self._purge=False
            
    def genUnitCircle(self,perturbation=1,boundary=2):
        """
        Sample points in fibonacci spiral on the unit circle
        
        Optional parameters:
            perturbation: type of perturbation (0 normal perturbation, 1 random perturbation), int
            boundary: type of boundary (0 jagged, >1 smooth)
            
        Return: None
        """
        shift = 1.0 if perturbation == 0 else self.N*np.random.random()

        ga = np.pi * (3.0 - np.sqrt(5.0))

        # Boundary points
        np_boundary = round(boundary * np.sqrt(self.N))

        self.dim=2
        self.ss = np.zeros((self.N,self.dim))
        self.pp = np.zeros((self.N,self.dim))
        j = 0
        for i in range(self.N):
            if i > self.N - (np_boundary + 1):
                r = 1.0
            else:
                r = np.sqrt((i + 0.5) / (self.N - 0.5 * (np_boundary + 1)))
            phi   = ga * (i + shift)
            self.ss[j,:] = np.array([r * np.cos(phi), r * np.sin(phi)])
            self.pp[j,:] = np.array([r,np.mod(phi,2*np.pi)])
            j += 1

    def genUnitHemisphere(self,perturbation=1,up=True):
        """
        Sample points in the unit hemisphere following fibonacci spiral
        
        Optional parameters:
            perturbation: type of perturbation (0 normal perturbation, 1 random perturbation), int
            up: side of hemisphere (True for upper hemisphere), boolean
            
        Return: None
        """
        n = 2 * self.N
        rn = range(self.N,n) if up else range(self.N) 

        shift = 1.0 if perturbation == 0 else n * np.random.random()

        ga = np.pi * (3.0 - np.sqrt(5.0))
        offset = 1.0 / self.N

        self.dim=3
        self.ss = np.zeros((self.N,self.dim))
        self.pp = np.zeros((self.N,self.dim))
        j = 0
        for i in rn:
            phi   = ga * ((i + shift) % n)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = ((i + 0.5) * offset) - 1.0
            theta=np.arccos(cos_theta)
            sin_theta = np.sqrt(1.0 - cos_theta*cos_theta)
            self.ss[j,:] = np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta])
            self.pp[j,:] = np.array([1,np.mod(phi,2*np.pi),np.pi/2-theta])
            j += 1

    def genUnitSphere(self,perturbation=1):
        """
        Sample points in the unit sphere following fibonacci spiral
        
        Optional parameters:
            perturbation: type of perturbation (0 normal perturbation, 1 random perturbation), int
            
        Return: None
        """

        shift = 1.0 if perturbation == 0 else self.N * np.random.random()

        ga = np.pi * (3.0 - np.sqrt(5.0))
        offset = 2.0 / self.N
        
        self.dim=3
        self.ss = np.zeros((self.N,self.dim))
        self.pp = np.zeros((self.N,self.dim))
        j = 0
        for i in range(self.N):
            phi   = ga * ((i + shift) % self.N)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = ((i + 0.5) * offset) - 1.0
            sin_theta = np.sqrt(1.0 - cos_theta*cos_theta)
            theta=np.arccos(cos_theta)            
            self.ss[j,:] = np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta])
            self.pp[j,:] = np.array([1,np.mod(phi,2*np.pi),np.pi/2-theta])
            j += 1

    def genCosineWeightedUnitHemisphere(self,perturbation=1):
        """
        Sample points in the unit hemisphere (with density weighted accordiging to cosine of colatitude)
        following fibonacci spiral
        
        Optional parameters:
            perturbation: type of perturbation (0 normal perturbation, 1 random perturbation), int
            
        Return: None
        """

        shift = 1.0 if perturbation == 0 else self.N * np.random.random()

        ga = np.pi * (3.0 - np.sqrt(5.0))

        self.dim=3
        self.ss = np.zeros((self.N,self.dim))
        self.pp = np.zeros((self.N,self.dim))
        j = 0
        for i in range(self.N):
            sin_theta = np.sqrt((i + 0.5) / (self.N - 0.5))
            cos_theta = np.sqrt(1.0 - sin_theta*sin_theta)
            phi   = ga * (i + shift)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            theta=np.arccos(cos_theta)                        
            self.ss[j,:] = np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta])
            self.pp[j,:] = np.array([1,np.mod(phi,2*np.pi),np.pi/2-theta])
            j += 1
    
    # Plot methods
    def _setEqualAspectRatio2D(self,xs,ys,alpha=1.5,delta=0.0):
        self.ax.set_aspect('equal')

        mn = np.array([xs.min(), ys.min()])
        mx = np.array([xs.max(), ys.max()])
        d = 0.5 * (mx - mn)
        c = mn + d
        d = alpha * np.max(d) + delta

        self.ax.set_xlim(c[0] - d, c[0] + d)
        self.ax.set_ylim(c[1] - d, c[1] + d)

    def _setEqualAspectRatio3D(self,xs,ys,zs,alpha=1.5,delta=0.0):
        self.ax.set_aspect('equal')

        mn = np.array([xs.min(), ys.min(), zs.min()])
        mx = np.array([xs.max(), ys.max(), zs.max()])
        d = 0.5 * (mx - mn)
        c = mn + d
        d = alpha * np.max(d) + delta

        self.ax.set_xlim(c[0] - d, c[0] + d)
        self.ax.set_ylim(c[1] - d, c[1] + d)
        self.ax.set_zlim(c[2] - d, c[2] + d)

    def updateCircleArgs(self,**args):
        self.cargs.update(args)
        
    def updateSphereArgs(self,**args):
        self.wargs.update(args)
        
    def plotSample(self,**args):
        sargs=dict(c='k',s=1.5)
        sargs.update(args)
        if self.dim==2:            
            self.fig,self.ax=plt.subplots(1,1)
            self.ax.scatter(self.ss[:,0],self.ss[:,1],**sargs)
            self.ax.add_patch(plt.Circle((0,0),1,**self.cargs))
            self._setEqualAspectRatio2D(self.ss[:,0],self.ss[:,1])
        else:
            self.fig=plt.figure()
            self.ax=plt.gca(projection='3d')
            self.ax.scatter(self.ss[:,0],self.ss[:,1],self.ss[:,2],**sargs)
            u,v=np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
            x=np.cos(u)*np.sin(v)
            y=np.sin(u)*np.sin(v)
            z=np.cos(v)
            self.ax.plot_wireframe(x,y,z,**self.wargs)
            self._setEqualAspectRatio3D(self.ss[:,0],self.ss[:,1],self.ss[:,2])

