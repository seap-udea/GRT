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

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

class Spice(object):

    #System constants
    _KERNELDIR=f"util/kernels"
    
    #Shapes 
    Ra=dict()
    Rb=dict()
    Rc=dict()
    f=dict()
    
    #Gravitational constants / masses
    #https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421_announcement.pdf
    Mu=dict(
        SSB=132712440040.944000*Const.km**3, #km^3/s^2
        SUN=132712440040.944000*Const.km**3, #km^3/s^2
        EARTH=398600.436233*Const.km**3, #km^3/s^2
        MOON=4902.800076*Const.km**3, #km^3/s^2
        EARTH_BARYCENTER=403503.236310*Const.km**3, #km^3/s^2
    )

    #Rotational Periods
    Prot=dict(
        SSB=27*Const.Day,
        SUN=27*Const.Day,
        EARTH=1*Const.Day,
        MOON=1*Const.SideralMonth,
    )
    
    #Reference Frames
    RF=dict(
        SSB="ECLIPJ2000",
        SUN="ECLIPJ2000",
        EARTH="ITRF93",
        MOON="IAU_MOON",
    )
    
    #Center
    Master=dict(
        SSB=[None,0],
        SUN=[None,0],
        EARTH=["SSB",Const.au],
        MOON=["EARTH",384000*Const.km]
    )
    
    def loadKernels():
        """
            Load Kernels
        """
        #Kernel sources: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
        kernels=[
            #Udates: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/
            "naif0012.tls",
            #Updates: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/
            "pck00010.tpc",
            #For updates and docs: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
            "de430.bsp",
            #Updates: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/
            "earth_720101_070426.bpc",
            "earth_070425_370426_predict.bpc",
            "earth_latest_high_prec_20190910.bpc",
            "moon_pa_de421_1900-2050.bpc",
            "moon_080317.tf"
            ]
        for kernel in kernels:
            spy.furnsh(f"{ROOTDIR}/data/{kernel}")
            
    def calcShape(objid):
        """
        Calculate shape of objetc objid.
        Parameters:
            objid: name of the object (eg. EARTH, MOON, etc.), string
        Return: None
        """
        try:
            Ra,Rb,Rc=spy.bodvrd(objid,"RADII",3)[1]
        except:
            Ra=Rb=Rc=1
            
        Spice.Ra[objid]=Ra*Const.km
        Spice.Rb[objid]=Rb*Const.km
        Spice.Rc[objid]=Rc*Const.km
        Spice.f[objid]=(Ra-Rc)/Ra
            
    def str2t(date):
        """
        Convert date from string TDB to TDB 
        Parameters:
            date: date string in TDB (eg. CCYY Mmm DD HH:HH:HH), string
        Returns:
            tdb: tdb, float, seconds since 2000 JAN 01 12:00:00 TDB.
        """
        et=spy.str2et(date)
        dt=spy.deltet(et,"ET")
        t=et-dt
        return t

class Body(object):
    """
    Define a body
    
    Atributes:
        objid: String with name of object (eg. MOON), string 
        refid: String with name of reference frame (eg. IAU_MOON), string
        P (optional): Rotational period, float, seconds
    """
    state=np.zeros(6)
    Tbod2ecl=np.zeros((3,3))
    
    def __init__(self,objid):
        self.id=objid
        
        if self.id is None:
            raise AssertionError("Body id is None")
        
        #Get geometrical, gravitational and physical properties
        self.rf=Spice.RF[self.id]
        self.master,self.amaster=Spice.Master[self.id]
        Spice.calcShape(self.id)
        self.Ra=Spice.Ra[self.id]
        self.Rb=Spice.Rb[self.id]
        self.Rc=Spice.Rc[self.id]
        self.f=Spice.f[self.id]
        self.mu=Spice.Mu[self.id]
        self.Prot=Spice.Prot[self.id]
        
        #Derivative
        if self.master is not None:
            self.rhill=self.amaster*(self.mu/(3*Spice.Mu[self.master]))**(1./3)
        else:
            self.rhill=1
            
    def updateBody(self,tdb):
        self.tdb=tdb
        self.state,self.tlight=spy.spkezr(self.id,tdb,"ECLIPJ2000","NONE","SSB")
        self.stateHelio=Const.transformState(self.state,[Const.km,Const.kms])
        self.Tbod2ecl=spy.pxform(self.rf,"ECLIPJ2000",tdb)
        self.Tecl2bod=np.linalg.inv(self.Tbod2ecl)
        
class Location(object):
    """
    Define a location on a scenario
    
    Atributtes:
        scenario: Scenario of the location, Scenario
        longitud: longitude, float, radians
        latitude: latitude, float, radians
        altitude: elevation over reference ellipsoid, float, m
    
    """
    
    def __init__(self,body,longitude,latitude,altitude):
        self.body=body
        self.lon=longitude
        self.lat=latitude
        self.alt=altitude
        
        self.posBody=spy.georec(self.lon,self.lat,self.alt,
                                self.body.Ra,self.body.f) 

        #Position of the location w.r.t. to itself (added for consistency)
        self.posLocal=np.zeros(3) 
        
        #Velocity local is the surface velocity due to planetary rotation: 2 pi rho/P
        rho=((self.posBody[:2]**2).sum())**0.5
        self.velLocal=np.array([0,+2*np.pi*rho/self.body.Prot,0])
        
        #Transformation matrix from local to body and viceversa
        uz=spy.surfnm(self.body.Ra,self.body.Rb,self.body.Rc,self.posBody)
        uy=spy.ucrss(np.array([0,0,1]),uz)
        ux=spy.ucrss(uz,uy)
        self.Tloc2bod=np.array(np.vstack((ux,uy,uz)).transpose().tolist())
        self.Tbod2loc=np.linalg.inv(self.Tloc2bod)
        
        #Velocity of the surface with respect to the inertial ref. frame of the body
        self.velBody=spy.mxv(self.Tloc2bod,self.velLocal)
    
    def updateLocation(self,tdb_moon):
        
        self.body.updateBody(tdb_moon)
        
        #Position of the location in the Ecliptic reference system
        self.posEcl=spy.mxv(self.body.Tbod2ecl,self.posBody)
        
        #Velocity of the location in the Ecliptic reference system
        self.velEcl=spy.mxv(self.body.Tbod2ecl,self.velBody)
    
    def vbod2loc(self,vBod):
        """
        Parameters:
            vBod: Vector in the body system, numpy array (3)
        Return:
            A: Azimuth (0,2 pi), float, radians
            h: Elevation (-pi,pi), float, radians
        """
        vLoc=spy.mxv(self.Tbod2loc,vBod)
        vimp,A,h=spy.reclat(vLoc)
        A=2*np.pi+A if A<0 else A
        return A,h,vimp

    def loc2vbod(self,A,h,v):
        """
        Express a vector in the direction A,h with magnitude v in the rotating
        reference frame of the central object of the location.
        
        Parameters:
            A: Azimuth (0,2 pi), float, radians
            h: Elevation (-pi,pi), float, radians
            v: Vector magnitude, (-infty,infty), float, (arbitrary)
               If v<0 then the vector points in the opposite direction of (A,h)
        Return:
            vBod: Velocity in the body-fixed system, np.array, km/s
        """
        vLoc=spy.latrec(v,A,h)
        vBod=spy.mxv(self.Tloc2bod,vLoc)
        return vBod
    
    def ecl2loc(self,eclon,eclat):
        """
        Parameters:
            eclon: Ecliptic longitude, float, radians
            eclat: Ecliptic latitude, float, radians
        Return:
            A: Azimuth (0,2 pi), float, radians
            h: Elevation (-pi,pi), float, radians
        
        NOTE: It requires to run previously the update method.
        """
        ecx,ecy,ecz=spy.latrec(1,eclon,eclat)
        x,y,z=spy.mxv(self.Tbod2loc,spy.mxv(self.body.Tecl2bod,[ecx,ecy,ecz]))
        r,A,h=spy.reclat([x,y,z])
        A=2*np.pi+A if A<0 else A
        return A,h

    def loc2ecl(self,A,h):
        """
        Parameters:
            A: Azimuth (0,2 pi), float, radians
            h: Elevation (-pi,pi), float, radians
        Return:
            eclon: Ecliptic longitude (0,2pi), float, radians
            eclat: Ecliptic latitude (-pi,pi), float, radians
            
        NOTE: It requires to run previously the update method.
        """
        x,y,z=spy.latrec(1,A,h)
        ecx,ecy,ecz=spy.mxv(self.body.Tbod2ecl,spy.mxv(self.Tloc2bod,[x,y,z]))
        r,eclon,eclat=spy.reclat([ecx,ecy,ecz])
        eclon=2*np.pi+eclon if eclon<0 else eclon
        return eclon,eclat

