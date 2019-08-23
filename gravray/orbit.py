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

# # GravRay Orbit Module

from gravray import *
from gravray.util import *
from gravray.spice import *

import pandas as pd
# https://pypi.org/project/quadpy/
import quadpy

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#################################################################################
#CLASS KEPLERIAN ORBIT
#################################################################################
class KeplerianOrbit(object):
    """
    A keplerian orbit
    
    Initialization attributes:
        mu: Gravitational parameter

    Secondary attributes:
        elements: cometary elements
            q: periapsis distance (0<q<inf), float, length
            e: eccentricity (e>=0), float
            i: inclination (0<i<pi), float, radians
            W: longitude of the ascending node (0<W<2pi), float, radians
            w: argument of the periapsis (0<w<2pi), float, radians
            M: mean anomaly (0<M<2pi), float, radians
    
        celements: classical elements:
            a: semimajor axis (-inf<a<inf), float, length
            e: eccentricity (e>=0), float
            i: inclination (0<i<pi), float, radians
            W: longitude of the ascending node (0<W<2pi), float, radians
            w: argument of the periapsis (0<w<2pi), float, radians
            M: mean anomaly (0<M<2pi), float, radians

        s: signatue (+1 ellipse,-1 hyperbola, 0 parabola)
            
        state: state vector (x,y,z,vx,vy,vz), numpy array (6), L,L,L,L/T,L/T,L/T
        
        uelements: unbound elements: A, E, I, O, W, m, numpy array (6), adimensional
            NOTE: Unbound elements are unconstrained (-inf,+inf)

        Derivative properties: Other elements:
            n: mean motion (n>0), float, 1/T
            ab: |a| (|a|>0), float, length
            eps: Eccentricity parameter, eps=sqrt[s(1-e**2)] (eps>=0), float
            b: semiminor axis, ab*eps
            cosE: "cosine" of the eccenric anomaly, cos(E) or cosh(H)
            sinE: "sine" of the eccenric anomaly, sin(E) or sinh(H)
            
    NOTE: Parabolas are not supported yet.

    """
    #Scales for unbound elements
    scales=[1.0,1.0,np.pi,np.pi,np.pi,np.pi]
    
    #Elements
    elements=None
    celements=None
    uelements=None
    
    #State
    state=np.zeros(6)
    
    #Secondary properties
    secondary=[]
    
    def __init__(self,mu):
        self.mu=mu
        
    def setElements(self,elements,t):
        """
        Set orbit by cometary elements.
        
        Parameters:
            elements: cometary elements, 
                q: periapsis distance (0<q<inf), float, length
                e: eccentricity (e>=0), float
                i: inclination (0<i<pi), float, radians
                W: longitude of the ascending node (0<W<2pi), float, radians
                w: argument of the periapsis (0<w<2pi), float, radians
                M: mean anomaly (0<M<2pi), float, radians
                
            t: time when the body has this mean anomaly M, float, time unit
        """
        self.elements=np.array(elements)
        if self.elements[1]==1:
            raise ValueError("Parabolas (e=1) are not supported yet.")
        
        self.t=t
        
        #Signature
        self._calcSignature()
        
        #Classical elements
        self.celements=np.copy(self.elements)
        self.celements[0]=self.elements[0]/(1-self.elements[1])
        
        #Compute state
        self.state=spy.conics(list(self.elements)+[t,self.mu],t)
        
        #Calculate secondary properties
        self._calcSecondary()
    
    def setUelements(self,uelements,t,scales=[1.0,1.0,np.pi,2*np.pi,2*np.pi,2*np.pi]):
        """
        Set orbit by unbound elements.
        
        Parameters:
            uelements: unbound elements Q,E,I,O,W,m                
            t: time when the body has this mean anomaly M, float, time unit
            
        Optional parameters:
            scales: maximum value of the orbital elements
        """
        self.uelements=np.array(uelements)
        elements=np.array([Util.inf2Fin(uelements[i],scales[i]) for i in range(6)])
        self.setElements(elements,t)
        
    def setState(self,state,t):
        """
        Set orbit by state.
        
        Parameters:
            state: state vector, (x,y,z,vx,vy,vz)
            t: time when the body has this mean anomaly M, float, time unit
        """
        self.state=np.array(state)
        self.t=t
        
        #Calculate elements corresponding to this state vector
        elements=spy.oscelt(self.state,t,self.mu)
        if elements[1]==1:
            raise ValueError("Parabolas (e=1) are not supported yet.")
  
        #Get cometary elements 
        self.elements=np.copy(elements[:6])
        
        #Orbital signature
        self._calcSignature()

        #Get classical elements
        self.celements=np.copy(self.elements)
        self.celements[0]=self.elements[0]/(1-self.elements[1])
        
        #Secondary properties
        self._calcSecondary()        
    
    def setCelements(self,celements,t):
        """
        Set orbit by classical elements.
        
        Parameters:
            celements: classical elements:
                a: semimajor axis (-inf<a<inf), float, length
                e: eccentricity (e>=0), float
                i: inclination (0<i<pi), float, radians
                W: longitude of the ascending node (0<W<2pi), float, radians
                w: argument of the periapsis (0<w<2pi), float, radians
                M: mean anomaly (0<M<2pi), float, radians
                
            t: time when the body has this mean anomaly M, float, time unit
        """
        self.celements=np.array(celements)
        if self.celements[1]==1:
            raise ValueError("Parabolas (e=1) are not supported yet.")

        a,e=self.celements[:2]
        if a*(1-e)<0:
            raise ValueError(f"Semimajor axis ({a}) and eccentricity ({e}) ar incompatible")

        #Compute cometary elements    
        self.elements=np.copy(self.celements)
        q=a*(1-e)
        self.elements[0]=q

        #Signature
        self._calcSignature()

        #Secondary properties
        self._calcSecondary()        
        
        self.t=t

    def calcUelements(self,maxvalues=[1.0,1.0,np.pi,2*np.pi,2*np.pi,2*np.pi]):
        if self.elements is None:
            raise ValueError("Cometary elements not set yet.  Use method setElements.")
        self.uelements=np.array([Util.fin2Inf(self.elements[i],maxvalues[i]) for i in range(6)])

    def calcStateByTime(self,t):
        """
        Update state and mean anomaly and secondaries at time t.
        
        Parameters:
            t: time, float, time units.

        Return:
            Md: Mean anomaly difference until t, float, radians.
            state: state at r, numpy float (6), L, L, L, L/T, L/T, L/T
        """
        
        #Update state using conics
        state=spy.conics(list(self.elements)+[self.t,self.mu],t)
        Md=self.secondary[0]*(t-self.t)
        return Md,t-self.t,state
    
    def calcStateByDistance(self,r,direction=+1):
        """
        Update state, mean anomaly and secondaries at distance r.
        
        Parameters:
            r: distance to central body
            
        Optional:
            direction: before (direction=-1) or after (direction=+1) present state, int
            
        Return:
            Md: Mean anomaly difference to reach r, float, radians.
            deltat: Time since present to reach r, float, time units.
            state: state at r, numpy float (6), L, L, L, L/T, L/T, L/T
        """
        a=self.celements[0]
        q=self.elements[0]
        e=self.elements[1]
        n=self.secondary[0]

        #True anomaly
        cosf=(q*(1+e)/r-1)/e
        if np.abs(cosf)<=1:
            fd=np.arccos(cosf)
        else:
            raise ValueError(f"Distance (r={r}) is not compatible with conic properties (q={q},a={a},e={e})")

        #Mean anomaly
        if e>1:
            Hd=2*np.arctanh(np.sqrt((e-1)/(e+1))*np.tan(fd/2))
            Md=e*np.sinh(Hd)-Hd
        else:
            Ed=2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(fd/2))
            Md=Ed-e*np.sin(Hd)

        #Time until reach r
        deltat=direction*Md/n

        #Heliocentric state
        state=spy.conics(self.elements.tolist()+[self.t,self.mu],self.t+deltat)

        return direction*Md,deltat,state
                
    def _calcSignature(self):
        #Signature
        self.s=+1 if self.elements[1]<1 else -1
    
    def _calcSecondary(self):

        #Get elements and state vector
        a,e,i,W,w,M=self.celements
        if e==1:
            raise ValueError("Parabolas (e=1) are not supported yet.")
        
        q=self.elements[0]
        x,y,z,vx,vy,vz=self.state
        mu=self.mu
        s=self.s
        r=(x**2+y**2+z**2)**0.5
        
        #Auxiliar
        cosi,sini=Angle.calcTrig(i)
        cosw,sinw=Angle.calcTrig(w)
        cosW,sinW=Angle.calcTrig(W)
        C=(cosw*sinW+sinw*cosi*cosW)
        D=(-sinw*sinW+cosw*cosi*cosW)
        
        #Secondary
        ab=np.abs(a)
        n=np.sqrt(mu/ab**3)
        eps=np.sqrt(s*(1-e**2))
        b=ab*eps
        if e>0:
            cosE=(1/e)*(1-r/a)
            sinE=(y-a*(cosE-e)*C)/(ab*eps*D)
        else:
            cosE,sinE=Angle.calcTrig(M)
        
        #Store secondary
        self.secondary=np.array([n,ab,eps,b,cosE,sinE])

#################################################################################
#CLASS JACOBIANS
#################################################################################
class Jacobians(object):
    """
    This abstract class contains useful methods for computing Jacobians.
    
    Attributes:
        None.
        
    Methods:
        computeJacobian: numerically compute jacobian.
    
    """
    def computeNumericalJacobian(jfun,x,dx,**args):
        """
        Computes numerically the Jacobian matrix of a multivariate function.
        
        Parameters:
            jfun: multivariate function with the prototype "def jfun(x,**args)", function
            x: indepedent variables, numpy array (N).
            dx: step size of independent variables, numpy array (N).
            **args: argument of the function
            
        Return:
            y: dependent variables, y=jfun(x,**args)
            Jyx: Jacobian matrix:
            
              Jif= [dy_1/dx_1,dy_1/dx_2,...,dy_1/dx_N,
                    dy_2/dx_1,dy_2/dx_2,...,dy_2/dx_N,
                                 . . . 
                    dy_N/dx_1,dy_N/dx_2,...,dy_N/dx_N,]
        """
        N=len(x)
        J=np.zeros((N,N))
        y=jfun(x,**args)
        for i in range(N):
            for j in range(N):
                pre=[x[k] for k in range(j)]
                pos=[x[k] for k in range(j+1,N)]
                yi=lambda t:jfun(pre+[t]+pos,**args)[i]
                dyidxj=(yi(x[j]+dx[j])-yi(x[j]-dx[j]))/(2*dx[j])
                J[i,j]=dyidxj
        return y,J

    def calcKeplerianJacobians(mu,celements,state):
        """
        Compute the Jacobian Matrix of the transformation from classical 
        orbital elements (a,e,i,w,W,M) to cartesian state vector (x,y,z,x',y',z').

        Return:

            Jc2k = [dx/da,dx/de,dx/di,dx/dw,dx/dW,dx/dM,
                    dy/da,dy/de,dy/di,dy/dw,dy/dW,dy/dM,
                    dz/da,dz/de,dz/di,dz/dw,dz/dW,dz/dM,
                    dx'/da,dx'/de,dx'/di,dx'/dw,dx'/dW,dx'/dM,
                    dy'/da,dy'/de',dy'/di,dy'/dw,dy'/dW,dy'/dM,
                    dz'/da,dz'/de,dz'/di',dz'/dw,dz'/dW,dz'/dM],

                    Numpy array 6x6, units compatible with mu and a.
        """
        a,e,i,W,w,M=celements
        q=a*(1-e)

        #Orbit signature
        if e<1:
            s=+1
        elif e>1:
            s=-1
        else:
            s=0
        
        #Trigonometric function
        cosi,sini=Angle.calcTrig(i)
        cosw,sinw=Angle.calcTrig(w)
        cosW,sinW=Angle.calcTrig(W)

        #Components of the rotation matrix
        A=(cosW*cosw-cosi*sinW*sinw);B=(-cosW*sinw-cosw*cosi*sinW)
        C=(cosw*sinW+sinw*cosi*cosW);D=(-sinw*sinW+cosw*cosi*cosW)
        F=sinw*sini;G=cosw*sini

        #Primary auxiliar variables
        ab=np.abs(a)
        n=np.sqrt(mu/ab**3)
        nu=n*a**2
        eps=np.sqrt(s*(1-e**2))

        #Get cartesian coordinates
        x,y,z,vx,vy,vz=state
        r=(x**2+y**2+z**2)**0.5
        nur=nu/r
        
        #Eccentric anomaly as obtained from indirect information
        #From the radial equation: r = a (1-e cos E)
        cosE=(1/e)*(1-r/a)

        #From the general equation for y
        #NOTE: This is the safest way to obtain sinE without the danger of singularities
        sinE=(y-a*(cosE-e)*C)/(ab*eps*D)

        #dX/da
        Ja=np.array([x/a,y/a,z/a,-vx/(2*a),-vy/(2*a),-vz/(2*a)])

        #dX/de
        dcosEde=-s*a*sinE**2/r
        dsinEde=a*cosE*sinE/r
        dnurde=(nu*a/r**2)*(cosE-(ab/r)*e*sinE**2)
        depsde=-s*e/eps

        drAde=a*(dcosEde-1)
        drBde=ab*(depsde*sinE+eps*dsinEde)

        dvAde=-(dnurde*sinE+nur*dsinEde)
        dvBde=(dnurde*eps*cosE+nur*depsde*cosE+nur*eps*dcosEde)

        Je=np.array([
            drAde*A+drBde*B,
            drAde*C+drBde*D,
            drAde*F+drBde*G,
            dvAde*A+dvBde*B,
            dvAde*C+dvBde*D,
            dvAde*F+dvBde*G,
        ])

        #dX/di
        Ji=np.array([z*sinW,-z*cosW,-x*sinW+y*cosW,vz*sinW,-vz*cosW,-vx*sinW+vy*cosW])

        #dX/dw
        Jw=np.array([-y*cosi-z*sini*cosW,x*cosi-z*sini*sinW,sini*(x*cosW+y*sinW),            -vy*cosi-vz*sini*cosW,vx*cosi-vz*sini*sinW,sini*(vx*cosW+vy*sinW)])

        #dX/dW
        JW=np.array([-y,x,0,-vy,vx,0])

        #dX/dM
        JM=np.concatenate(((ab**3/mu)**0.5*np.array([vx,vy,vz]),
                           (mu*ab**3)**0.5*np.array([-x/r**3,-y/r**3,-z/r**3])))

        #Jacobian
        Jck=np.array([Ja,Je,Ji,JW,Jw,JM]).transpose()

        return Jck

    def calcDetMapJacobian(elements,scales):
        """
        Parameters:
            epsilon: bound elements, numpy array (N)
            scales: scales for the bound elements ()

        Return:

            Jif= [dE_1/de_1,        0,        0,...,        0,
                          0,dE_2/de_2,        0,...,        0,
                          0,        0,dE_2/de_2,...,        0,
                                 . . . 
                          0,        0,        0,...,dE_N/de_N]

            where dE/de = (1/s) /[x(1-x)] and x = e/s.
        """
        JEe=np.identity(6)
        JeE=np.identity(6)

        detJEe=1
        detJeE=1
        for i,eps in enumerate(elements):
            x=eps/scales[i]
            detJEe*=(1/scales[i])/(x*(1-x))
            
        detJeE=1/detJEe
        return detJEe,detJeE
    
    def calcImpactJacobian(body,Rimp,state):
        """
        Compute the Jacobian Matrix of the transformation from local impact conditions 
        (lon,lat,alt,A,h,v) to cartesian state vector (x,y,z,x',y',z') (in the body reference frame).

        Parameters:
            Rimp: Impact vector
                lon: Geographic longitude (0,2pi), float, radians
                lat: Geographic latitude (0,2pi), float, radians
                alt: Altitude over the ellipsoid (0,inf), float, km
                A: Azimuth (0,2pi), float, radians
                h: Elevation (-pi/2,pi/2), float, radians
                v: Impact speed (-inf,inf), float, km/s (negative if it is impacting)
            state: State vector (x,y,z,x',y',z'), numpy array (6)

        Return:

            Jcl = [dx/dlon,dx/dlat,dx/dalt,dx/dA,dx/dh,dx/dv,
                   dy/dlon,dy/dlat,dy/dalt,dy/dA,dy/dh,dy/dv,
                   dz/dlon,dz/dlat,dz/dalt,dz/dA,dz/dh,dz/dv,
                   dx'/dlon,dx'/dlat,dx'/dalt,dx'/dA,dx'/dh,dx'/dv,
                   dy'/dlon,dy'/dlat,dy'/dalt,dy'/dA,dy'/dh,dy'/dv,
                   dz'/dlon,dz'/dlat,dz'/dalt,dz'/dA,dz'/dh,dz'/dv],

                Numpy 6x6 array.
        """
        #Local to rotating
        lon,lat,alt,A,h,vimp=Rimp
        x,y,z,vx,vy,vz=state
        
        coslon,sinlon=Angle.calcTrig(lon)
        coslat,sinlat=Angle.calcTrig(lat)
        cosA,sinA=Angle.calcTrig(A)
        cosh,sinh=Angle.calcTrig(h)
        
        P=body.Prot
        a=body.Ra
        b=body.Rc
        Tbod2ecl=body.Tbod2ecl
        
        #Auxiliar
        fr=2*np.pi*np.sqrt(x**2+y**2)/(P*vimp)
        N=a**2/np.sqrt(a**2*coslat**2+b**2*sinlat**2)
        n2=(2*np.pi/P)**2

        #dX/dlon:
        Jlon=np.array([-y,x,0,-vy,vx,0])

        #dX/dlat:
        dxdlat=(a**2-b**2)*coslat*sinlat*N**3/a**4*coslat*coslon-(N+alt)*sinlat*coslon
        dydlat=(a**2-b**2)*coslat*sinlat*N**3/a**4*coslat*sinlon-(N+alt)*sinlat*sinlon
        Jlat=np.array([
            dxdlat,
            dydlat,
            b**2*(a**2-b**2)*coslat*sinlat*N**3/a**6*sinlat+(b**2*N/a**2+alt)*coslat,
            -vimp*cosh*cosA*coslat*coslon-n2*sinlon/(fr*vimp)*(x*dxdlat+y*dydlat)-vimp*sinh*sinlat*coslon,
            -vimp*cosh*cosA*coslat*sinlon+n2*coslon/(fr*vimp)*(x*dxdlat+y*dydlat)-vimp*sinh*sinlat*sinlon,
            vimp*(-cosh*cosA*sinlat+sinh*coslat)
        ])

        #dX/dalt:
        Jalt=np.array([
            coslat*coslon,coslat*sinlon,sinlat,
            -n2*sinlon/(fr*vimp)*(x*coslat*coslon+y*coslat*sinlon),
            +n2*coslon/(fr*vimp)*(x*coslat*coslon+y*coslat*sinlon),
            0
        ])

        #dX/dA:
        JA=np.array([0,0,0,
            vimp*(cosh*sinA*sinlat*coslon-cosh*cosA*sinlon),
            vimp*(cosh*sinA*sinlat*sinlon+cosh*cosA*coslon),
            -vimp*cosh*sinA*coslat,
           ])

        #dX/dh:
        Jh=np.array([0,0,0,
            vimp*(sinh*cosA*sinlat*coslon+sinh*sinA*sinlon+cosh*coslat*coslon),
            vimp*(sinh*cosA*sinlat*sinlon-sinh*sinA*coslon+cosh*coslat*sinlon),
            vimp*(-sinh*cosA*coslat+cosh*sinlat),
            ])

        #dX/dvimp:
        Jv=np.array([0,0,0,vx/vimp+sinlon*fr,vy/vimp-coslon*fr,vz/vimp])

        Jcl=np.array([Jlon,Jlat,Jalt,JA,Jh,Jv]).transpose()
        Jel=np.zeros_like(Jcl)
        for i in range(6):
            Jel[:3,i]=spy.mxv(Tbod2ecl,Jcl[:3,i])
            Jel[3:,i]=spy.mxv(Tbod2ecl,Jcl[3:,i])
        return Jcl,Jel

class GrtRay(object):
    """
    A ray in a GRT analysis
    
    Initialization attributes:
        location: Location of the ray, object of class Location
        azimuth: Azimuth of the ray (0,2pi), float, radians
        elevation: Elevation of the ray (-pi/2,pi/2), float, radians
        speed: speed of the ray at the location (in the rotating reference frame), float, km/s
        
            NOTE: It is VERY importante to take into account that if a particle is COMING from A,h, 
                  you need to specify its velocity as (A,h,-speed) or (360-A,-h,speed).

    Secondary attributes:r
        raydir: sign of the directior of the ray (-1 towards the Earth, +1 from the Earth), int
        body: Central body where the ray starts.
        
    """
    
    def __init__(self,location,azimuth,elevation,speed):

        #Initialization attributes
        self.A=azimuth
        self.h=elevation
        self.vimp=speed

        #Sign of the ray
        self.raydir=np.sign(self.vimp)
                
        #Location
        self.location=deepcopy(location)        
        self.body=deepcopy(location.body)
        self.body.Tbod2ecl=None
        self.body.Tecl2bod=None
        
        #Impact vector
        self.Rimp=np.array([self.location.lon,self.location.lat,self.location.alt,self.A,self.h,self.vimp])

        #Instantiate masters
        master=self.body.master
        self.masters=dict()
        while master is not None:
            self.masters[master]=Body(master)
            master=self.masters[master].master

        #Body-centric state vector in the body axis
        self.velLoc=spy.latrec(self.vimp,self.A,self.h)
        self.posBody=np.copy(self.location.posBody)
        self.velBody=spy.mxv(self.location.Tloc2bod,self.velLoc)+self.location.velBody
        self.stateBody=np.concatenate((self.posBody,self.velBody))
        
        #Initialize in None
        self.states=None

    def updateRay(self,tdb):
        self.tdb=tdb
        self.location.updateLocation(tdb)
        self.body.updateBody(tdb)
        
        #Body-centric state vector in the ecliptic axis
        self.posEcl=np.copy(self.location.posEcl)
        self.velEcl=spy.mxv(self.body.Tbod2ecl,self.velBody)
        self.stateEcl=np.concatenate((self.posEcl,self.velEcl))
        self.stateHelio=self.stateEcl+self.body.stateHelio

    def propagateRay(self,tdb):
        """
        Propagate ray starting at time tdb.
        
        Parameters:
            tdb: time of impact (in ephemeris time), float, seconds
        """

        #Update ray
        self.updateRay(tdb)
        
        #Compute escape state
        self.states=[]
        self.states+=[(tdb,self.body.mu,None,self.stateEcl,None,None,f"Surface of {self.body.id}")]
        et,state=self.propagateEscapeState(self.stateEcl,et=tdb,direction=self.raydir)
        
        #Store final state and orbital elements
        self.terminal=KeplerianOrbit(Spice.Mu["SSB"])
        self.terminal.setState(state,et)
        self.states+=[(et,Spice.Mu["SSB"],self.terminal,state,None,None,f"Final escape")]
    
    def propagateEscapeState(self,state,et=0,direction=1):
        """
        Compute the transformation from the position w.r.t. to the central body of the ray
        to the position w.r.t. the master of the central body
        
        Parameters:
            state: Initial state of a ray (w.r.t. body in ecliptic reference frame), 
                   numpy array (6), [L,L,L,L/T,L/T,L/T]
            et: Initial time (in ephemeris time), float, seconds
            
        Return:
            et: Final time of propagation
            state: Final state of a rey (w.r.t. SSB), numpy array (6), [L,L,L,L/T,L/T,L/T]
        """
        #Set initial orbit
        body=self.body
        helio=self.body.stateHelio
        statehelio=state=state+helio
        while body.master is not None:
            #State with respect to body
            state-=helio
            #Escape state
            q,e,i,Omega,omega,M,et,mu=spy.oscelt(state,et,body.mu)
            if e<1:
                raise AssertionError(f"The object has collided against {body.id} with e={e} (Rimp = {self.Rimp}) ")        
            orbit=KeplerianOrbit(body.mu)
            orbit.setElements([q,e,i,Omega,omega,M],et)
            Md,deltat,state=orbit.calcStateByDistance(body.rhill,direction=direction)
            helio=body.calcState(et+deltat)
            statehelio=state+helio 
            #Position with respect to new body
            et+=deltat
            state=statehelio
            body=self.masters[body.master]
            helio=body.calcState(et)

        return et,state

    def calcJacobianDeterminant(self):
        """
        Compute Jacobian determinant
        """
        #Jxi := dXgeo/dRimp
        Rimp=self.Rimp
        state=self.stateBody
        Jxi,Jix=Jacobians.calcImpactJacobian(self.body,Rimp,state)
        detJxi=np.linalg.det(Jxi)
            
        #Jhx := dXgeo/dXhel
        Xbody2Xhel=lambda X:self.propagateEscapeState(X,et=self.states[0][0],direction=-1)[-1]
        X=self.states[0][3]
        dX=np.abs((X+1e-5)*1e-5)
        y,Jhx=Jacobians.computeNumericalJacobian(Xbody2Xhel,X,dX)
        detJhx=np.linalg.det(Jhx)
        
        #Jeh := dehel/dXhel
        hel_et,hel_mu,hel_orbit,hel_state,hel_helio,hel_statehelio,hel_name=self.states[-1]
        Jhe=Jacobians.calcKeplerianJacobians(hel_mu,hel_orbit.celements,hel_orbit.state)
        Jeh=np.linalg.inv(Jhe)
        detJeh=np.linalg.det(Jeh)
        
        #|Jei| := |Jeh| |Jhx| |Jxi|
        detJei=detJeh*detJhx*detJxi
        
        return detJei
    
    def packRay(self):
        import pandas as pd
        
        #Ray dataframe columns
        columns=["et","lon","lat","alt","A","h","v",
                 "ximp","yimp","zimp","vximp","vyimp","vzimp",
                 "xhel","yhel","zhel","vxhel","vyhel","vzhel",
                 "q","e","i","W","w","M",
                 "a","n"
                ]

        #Extract info
        Rimp=self.Rimp*np.array([Angle.Rad,Angle.Rad,1,Angle.Rad,Angle.Rad,1])
        
        try:
            stateimp=self.states[0]
            et=stateimp[0]
            ximp=self.stateHelio
            xhel=self.states[-1][3]
            elements=Util.transformElements(self.terminal.elements,[1/Const.au,Angle.Rad])
            celements=np.array([self.terminal.celements[0]/Const.au,self.terminal.secondary[0]])
        except:
            et=self.tdb
            ximp=self.stateHelio
            xhel=stateimp=elements=np.zeros(6)
            celements=np.zeros(2)

        #Pack
        raydf=pd.DataFrame([np.concatenate(([et],Rimp,ximp,xhel,elements,celements))],columns=columns)
        return raydf

class GrtProb(object):
    """
    Class used to compute the impact probability
    """
    
    def __init__(self,t,body,population,verbose=False):

        #Primary
        self.body=body
        self.tdb=t       
        self.population=population
        
        #Behavior
        self.verbose=verbose
        
        #Secondary
        self.location=None
        
    def setLocation(self,lon,lat,alt):
        self.location=Location(self.body,lon,lat,alt)
    
    def calcPdir(self,Rdir):
        """
        Compute the resulting impact probability at a given site and 
        time and with objects coming from a given population.

        Parameters:

            Rdir: impact vector (azimuth, elevation and speed), numpy array (3), (rad, rad, m/s)

        Return:

            pimp: Impact probability, p(R_i) = p(E_h) |det JER|.
        """
        self.Rdir=Rdir
        A,h,v=Rdir
        ray=GrtRay(self.location,A,h,v)
        ray.updateRay(self.tdb)
        try:
            ray.propagateRay(self.tdb)
            ray.terminal.elements[3:]=np.mod(ray.terminal.elements[3:],2*np.pi)
            self.elements=Util.transformElements(ray.terminal.elements,[1/Const.au,Angle.Rad])
            
            if self.verbose:
                print(f"Impact conditions: np.array([[{A}],[{h}],[{v}]])")
                print(f"Terminal elements:",
                      Util.transformElements(ray.terminal.elements,[1/Const.au,Angle.Rad]))
                try:
                    ray.terminal.calcUelements([Const.aphelion,1,np.pi,2*np.pi,2*np.pi,2*np.pi])
                    locelements=ray.terminal.uelements
                    print("Unbound terminal elements:",locelements)
                except:
                    print("Hyperbolic orbit")

            #Check if orbit is bound
            if ray.terminal.elements[1]<1:
                #J = |dEhel/dRimp| = |dchel/dRimp| x |dehel/dchel| x |dEhel/dehel| 
                detJ=ray.calcJacobianDeterminant()*                     (1-ray.terminal.elements[1])*                     Jacobians.calcDetMapJacobian(ray.terminal.elements,
                                                  [Const.aphelion,1,np.pi,2*np.pi,2*np.pi,2*np.pi])[0]
                ray.terminal.calcUelements(maxvalues=[Const.aphelion,1,np.pi,2*np.pi,2*np.pi,2*np.pi])
                x=ray.terminal.uelements[:5]
                ph=self.population.pdf(x)
            #If orbit is not bound probability is zero
            else:
                ph=0
                detJ=0

        except AssertionError as e:
            if self.verbose:
                print("Error in ray:",e)
            self.elements=-1*np.ones(6)
            detJ=0
            ph=0
        del ray
        self.ph=ph
        self.detJ=detJ
        return ph*np.abs(detJ)
    
    def calcPsky(self,Rsky):
        N=Rsky.shape[1]
        p=np.zeros(N)
        self.phs=np.zeros(N)
        self.detJs=np.zeros(N)
        self.selements=np.zeros((N,6))
        self.Rsky=Rsky.transpose()
        for i in range(N):
            p[i]=self.calcPdir(Rsky[:,i])
            self.phs[i]=self.ph
            self.detJs[i]=self.detJ
            self.selements[i,:]=self.elements
        return p
    
    def calcPimp(self,Rimp):
        """
        Compute the resulting impact probability for a given impact vector, 
        time and with objects coming from a given population.

        Parameters:

            Rimp: impact matrix (longitude, latitude, height, azimuth, elevation, speed), 
                  ie. rows are properties and columns are different values of properties.
                  numpy array (6xN), (rad, rad, m, rad, rad, m/s) x N

        Return:

            p: Impact probability p(R_i) = p(E_h) |det JER|, numpy array (N)
        """
        lon,lat,alt=Rimp[:3]
        Rdir=Rimp[3:]
        loc=Location(self.body,lon,lat,alt)
        self.location=loc
        p=self.calcPdir(Rdir)
        del loc
        return p    
    
    def calcPimpVector(self,Rimp):
        """
        Compute the resulting impact probability for a given impact vector, 
        time and with objects coming from a given population.

        Parameters:

            Rimp: impact matrix (longitude, latitude, height, azimuth, elevation, speed), 
                  ie. rows are properties and columns are different values of properties.
                  numpy array (6xN), (rad, rad, m, rad, rad, m/s) x N

        Return:

            p: Impact probability p(R_i) = p(E_h) |det JER|, numpy array (N)
        """
        N=Rimp.shape[1]
        p=np.zeros(N)
        self.phs=np.zeros(N)
        self.detJs=np.zeros(N)
        self.selements=np.zeros((N,6))
        self.Rimp=Rimp.transpose()
        for i in range(N):
            lon,lat,alt=Rimp[:3,i]
            Rdir=Rimp[3:,i]
            if self.verbose:
                print(f"Location: lon = {lon*Angle.Rad}, lat = {lat*Angle.Rad}, alt = {alt/Const.km}")
                print(f"Direction: A = {Rdir[0]*Angle.Rad}, h = {Rdir[1]*Angle.Rad}, v = {Rdir[2]/Const.km}")
            loc=Location(self.body,lon,lat,alt)
            self.location=loc
            p[i]=self.calcPdir(Rdir)
            self.phs[i]=self.ph
            self.detJs[i]=self.detJ
            self.selements[i,:]=self.elements
            del loc
        return p

