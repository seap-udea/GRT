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

from gravray import *
from gravray.util import *
from gravray.spice import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

class KeplerianOrbit(object):
    """
    A keplerian orbit
    
    Input attributes:
        mu: Gravitational parameter

    Other attributes:
        elements: cometary elements
            a: semimajor axis (-inf<a<inf), float, length
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

        s: signatue (+1 ellipse,-1 hyperbola, 0 parabolla)
            
        state: state vector (x,y,z,vx,vy,vz), numpy array (6), L,L,L,L/T,L/T,L/T
        
        uelements: unbound elements: A, E, I, O, W, M

        Derivative properties: Other elements:
            n: mean motion (n>0), float, 1/T
            ab: |a| (|a|>0), float, length
            eps: Eccentricity parameter, eps=sqrt[s(1-e**2)] (eps>=0), float
            b: semiminor axis, ab*eps
            cosE: "cosine" of the eccenric anomaly, cos(E) or cosh(H)
            sinE: "sine" of the eccenric anomaly, sin(E) or sinh(H)

    """
    s=0
    scales=[1,1,np.pi,np.pi,np.pi,np.pi]
    elements=np.zeros(6)
    celements=np.zeros(6)
    uelements=np.zeros(6)
    state=np.zeros(6)
    derivatives=[]
    
    def __init__(self,mu):
        self.mu=mu
        
    def setElements(self,elements,t):
        self.elements=elements
        self.t=t
        if self.elements[1]>1:
            self.s=-1
        else:
            self.s=+1
        self.state=spy.conics(list(self.elements)+[t,self.mu],t)
        np.copyto(self.celements,self.elements)
        self.celements[0]=self.elements[0]/(1-self.elements[1])
        self.calcDerivatives()
    
    def setUelements(self,uelements,t,maxvalues=[1.0,1.0,np.pi,2*np.pi,2*np.pi,2*np.pi]):
        elements=np.array([Util.inf2Fin(uelements[i],maxvalues[i]) for i in range(6)])
        self.setElements(elements,t)
        
    def setState(self,state,t):
        self.state=state
        self.t=t
        elements=spy.oscelt(self.state,t,self.mu)
        self.elements=elements[:6]
        np.copyto(self.celements,self.elements)
        self.celements[0]=self.elements[0]/(1-self.elements[1])
        if self.elements[1]>1:
            self.s=-1
        else:
            self.s=+1
        self.calcDerivatives()        
        
    def updateState(self,t):
        self.state=spy.conics(list(self.elements)+[self.t,self.mu],t)
        #Update derivatives
        self.calcDerivatives()
        #Update M (it can be improved): M = s (E - e sinE) (where sinE is sinhE in case of e>1)
        self.celements[-1]=self.elements[-1]=spy.oscelt(self.state,t,self.mu)[5]
        self.t=t
        
    def calcDerivatives(self):

        #Get elements and state vector
        a,e,i,W,w,M=self.celements
        q=self.elements[0]
        x,y,z,vx,vy,vz=self.state
        mu=self.mu
        s=self.s
        r=(x**2+y**2+z**2)**0.5

        #Auxiliar
        cosi,sini=Angle.calcTrig(i)
        cosw,sinw=Angle.calcTrig(w)
        cosW,sinW=Angle.calcTrig(W)
        C=(cosw*sinW+sinw*cosi*cosW);D=(-sinw*sinW+cosw*cosi*cosW)
        
        #Derivatives
        ab=np.abs(a)
        n=np.sqrt(mu/ab**3)
        eps=np.sqrt(s*(1-e**2))
        b=ab*eps
        cosE=(1/e)*(1-r/a)
        sinE=(y-a*(cosE-e)*C)/(ab*eps*D)
        
        self.derivatives=np.array([n,ab,eps,b,cosE,sinE])
    
    def calcUelements(self,maxvalues=[1.0,1.0,np.pi,2*np.pi,2*np.pi,2*np.pi]):
        self.uelements=np.array([Util.fin2Inf(self.elements[i],maxvalues[i]) for i in range(6)])
    
    
    def calcJacobians(self):
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
        a,e,i,W,w,M=self.celements
        q=self.elements[0]
        mu=self.mu
        s=self.s
        
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
        x,y,z,vx,vy,vz=self.state
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
        JM=ab**1.5*np.array([vx,vy,vz,-x/r**3,-y/r**3,-z/r**3])

        #Jacobian
        self.Jck=np.array([Ja,Je,Ji,Jw,JW,JM]).transpose()
        self.Jkc=np.linalg.inv(self.Jck)
    
    def calcJacobiansMap(self):
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
        self.JEe=np.identity(6)
        self.JeE=np.identity(6)
        for i,eps in enumerate(self.elements):
            x=eps/self.scales[i]
            self.JEe[i,i]=(1/self.scales[i])/(x*(1-x))
            self.JeE[i,i]=1/self.JEe[i,i]

class GrtRay(object):
    """
    A ray in a GRT analysis
    
    Input attributes:
        location: Location of the ray, object of class Location
        azimuth: Azimuth (0,2pi), float, radians
        elevation: Elevation (-pi/2,pi/2), float, radians
        speed: speed of the ray at the location (in the rotating reference frame), float, km/s
        
        NOTE: It is VERY importante to take into account that if a particle is COMING from A,h, 
              you need to specify its velocity as (A,h,-speed) or (360-A,-h,speed).
              
    Other attributes:
        scenario: Scenario where the ray propagates.
        body: Central body where the ray starts.
        
    """
    
    def __init__(self,location,azimuth,elevation,speed):
        
        #Attributes
        self.location=location
        self.body=self.location.body
        self.A=azimuth
        self.h=elevation
        self.vimp=speed
        
        #Instantiate masters
        master=self.body.master
        self.masters=dict()
        while master is not None:
            self.masters[master]=Body(master)
            master=self.masters[master].master

        #Body-centric state vector in the body axis
        self.velLoc=spy.latrec(self.vimp,self.A,self.h)
        self.velBody=spy.mxv(self.location.Tloc2bod,self.velLoc)+self.location.velBody
        self.stateBody=np.concatenate((self.location.posBody,self.velBody))

    def updateRay(self,tdb):
        self.tdb=tdb
        self.location.updateLocation(tdb)
        
        #Body-centric state vector in the ecliptic axis
        self.velEcl=spy.mxv(self.body.Tbod2ecl,self.velBody)
        self.stateEcl=np.concatenate((self.location.posEcl,self.velEcl))
        
        #Jacobian of the transformation (lon,lat,alt,A,h,vimp)->(x,y,z,x',y',z')
        #self.Jxgeo2rimp
    
    def calcJacobiansBody(self):
        """
        Compute the Jacobian Matrix of the transformation from 
        local impact conditions (lon,lat,alt,A,h,v) to cartesian state vector (x,y,z,x',y',z') 
        (in the body reference frame).

        Parameters:
            lon: Geographic longitude (0,2pi), float, radians
            lat: Geographic latitude (0,2pi), float, radians
            alt: Altitude over the ellipsoid (0,inf), float, km
            A: Azimuth (0,2pi), float, radians
            h: Elevation (-pi/2,pi/2), float, radians
            v: Impact speed (-inf,inf), float, km/s (negative if it is impacting)

        Return:

            Jc2l = [dx/dlon,dx/dlat,dx/dalt,dx/dA,dx/dh,dx/dv,
                    dy/dlon,dy/dlat,dy/dalt,dy/dA,dy/dh,dy/dv,
                    dz/dlon,dz/dlat,dz/dalt,dz/dA,dz/dh,dz/dv,
                    dx'/dlon,dx'/dlat,dx'/dalt,dx'/dA,dx'/dh,dx'/dv,
                    dy'/dlon,dy'/dlat,dy'/dalt,dy'/dA,dy'/dh,dy'/dv,
                    dz'/dlon,dz'/dlat,dz'/dalt,dz'/dA,dz'/dh,dz'/dv],

                    Numpy 6x6 array.
        """

        #Local to rotating
        lon,lat,alt,A,h,vimp=self.location.lon,self.location.lat,self.location.alt,self.A,self.h,self.vimp
        x,y,z,vx,vy,vz=self.stateBody
        
        coslon,sinlon=Angle.calcTrig(lon)
        coslat,sinlat=Angle.calcTrig(lat)
        cosA,sinA=Angle.calcTrig(A)
        cosh,sinh=Angle.calcTrig(h)
        
        P=self.location.body.Prot
        a=self.location.body.Ra
        b=self.location.body.Rc
        
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

        self.Jcl=np.array([Jlon,Jlat,Jalt,JA,Jh,Jv]).transpose()
        self.Jlc=np.linalg.inv(self.Jcl)
    
    def propagateRay(self):
        
        state=np.zeros(6)

        body=self.body
        state=self.stateEcl+body.stateHelio
        et=self.tdb
        
        self.conics=[]
        while body.master is not None:

            #State w.r.t. to body
            body.updateBody(et)
            state=state-body.stateHelio
        
            #Get object-centric elements
            q,e,i,Omega,omega,Mo,et,mu=spy.oscelt(state,et,body.mu)
            a=q/(1-e)
            n=np.sqrt(body.mu/np.abs(a)**3)
            self.conics+=[[q,e,i,Omega,omega,Mo,body.mu]]
            
            #hill
            etp=et-Mo/n
            fd=np.arccos((q*(1+e)/body.rhill-1)/e)
            Hd=2*np.arctanh(np.sqrt((e-1)/(e+1))*np.tan(fd/2))
            Md=e*np.sinh(Hd)-Hd
            deltat=Md/n
            
            #Update body position
            body.updateBody(etp-deltat)

            #Heliocentric conic:
            hillstate=spy.conics([q,e,i,Omega,omega,Mo,et,body.mu],etp-deltat)
            self.conics+=[[q,e,i,Omega,omega,-Md,body.mu]]
            
            #Next conic
            et=etp-deltat
            state=hillstate+body.stateHelio

            body=self.masters[body.master]
        
        self.terminal=KeplerianOrbit(Spice.Mu["SSB"])
        self.terminal.setState(state,et)
        self.conics+=[list(self.terminal.elements)+[Spice.Mu["SSB"]]]

