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

# # GravRay SPICE Module

from gravray import *
from gravray.util import *
import re

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

#################################################################################
#SPICE data structure
#################################################################################
class Spice(object):
    """
    This abstract class contains methods Spice.

    Attributes:
        None.

    Data:
        
    Methods:
    
    """

    #System constants
    _KERNELDIR=f"util/kernels"
    
    #Kernel sources: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
    Kernels=[
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
        SSB=0,
        SUN=0,
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
        MOON=["EARTH",385000 *Const.km]
    )
    
    #Shapes 
    Ra=dict()
    Rb=dict()
    Rc=dict()
    f=dict()
    RH=dict()
    
    def loadKernels():
        """
        Load Kernels
        """
        for kernel in Spice.Kernels:
            spy.furnsh(f"{ROOTDIR}/data/{kernel}")
            
    def calcHillRadius(objid):
        """
        Calculate the Hill radius of object it.
        
        Parameters:
            objid: name of the object (eg. EARTH, MOON, etc.), string
        
        Return: None
        """
        try:
            master=Spice.Master[objid]
        except Exception as e:
            errorMsg(e,f"Object {objid} is not registered.")
            raise
        
        if master[0] is None:
            Spice.RH[objid]=0
            return
        a=master[1]
        Mmaster=Spice.Mu[master[0]]
        Mbody=Spice.Mu[objid]            
        Spice.RH[objid]=a*(Mbody/(3*Mmaster))**(1./3)
        
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
            
        Example:
            tdb=Spice.str2t("2000 JAN 01 12:00:00")
            tdb must be zero.            
        """
        et=spy.str2et(date)
        dt=spy.deltet(et,"ET")
        t=et-dt
        return t
    
    def str2tdb(date):
        """
        Convert date from string to TDB 

        Parameters:
            date: date string (eg. CCYY Mmm DD HH:HH:HH [TDB[+/-N]|UTC[+/-N]]), string

        Returns:
            tdb: tdb, float, seconds since 2000 JAN 01 12:00:00 TDB.

        Example:
            tdb=Spice.str2t("2000 JAN 01 12:00:00 TDB"), tdb = 0

            tdb=Spice.str2t("2000 JAN 01 12:00:00 UTC"), tdb = 64.2 

            tdb=Spice.str2t("2000 JAN 01 12:00:00"), tdb = 64.2 

            tdb=Spice.str2t("2000 JAN 01 12:00:00 TDB-5"), tdb = 18000 
        """
        parts=date.split(" ")
        ttype="UTC"
        zone=parts[-1]

        if re.search(":",parts[-1]) is None:
            if re.search("TDB",zone):ttype="TDB"
            zone=zone.replace("TDB","UTC")
        dstring=" ".join(parts[:-1])+f" {zone}"

        if ttype=="TDB":
            et=spy.str2et(dstring)
            dt=spy.deltet(et,"ET")
            t=et-dt
        else:
            t=spy.str2et(dstring)

        return t
    
    def zappalaDistance(E1,E2):
        """                                                                                                                                                                              
        Zappala (1990), Nervorny & Vokrouhlicky (2006)
        am=(a+at)/2
        d2c=1/np.sqrt(am)*(ka*((at-a)/am)**2+ke*(et-e)**2+ki*(sinit-np.sin(i*DEG))**2+kO*(Omega-Ot)**2+kw*(omega-ot)**2)
        Parameters:
            E1: Elements 1, np.array(5), [q(UL),e,i(rad),W(rad),w(rad)]
            E2: Elements 1, np.array(5), [q(UL),e,i(rad),W(rad),w(rad)]
        Return:
            DZ: Zappala distance (when [q]=AU, DZ<0.1 is a good match), float
        """
        #Coefficients
        ka=5./4
        ke=ki=2
        kw=kW=1e-4
        #Elements
        q1,e1,i1,W1,w1=E1[:5]
        q2,e2,i2,W2,w2=E2[:5]
        #Derived elements
        sini1=np.sin(i1)
        sini2=np.sin(i2)

        if e1!=1:
            a1=np.abs(q1/(1-e1))
        else:
            a1=q1

        if e2!=1:
            a2=np.abs(q2/(1-e2))
        else:
            a2=q2

        am=(a1+a2)/2
        anm=1/np.sqrt(np.abs(am))
        varpi1=W1+w1
        varpi2=W2+w2
        #Zappala metric (Zuluaga & Sucerquia, 2018)
        DZ=anm*(ka*(a1-am)**2/am**2+ke*(e1-e2)**2+ki*(sini1-sini2)**2+kW*(W1-W2)**2+kw*(varpi1-varpi2)**2)**0.5

        return DZ

#################################################################################
#Body Class
#################################################################################
class Body(object):
    """
    Class describing an astronomical body, including its position and velocity in space.
    
    Initialization atributes:
        objid: String with name of object (eg. MOON), string 
        
    Secondary attributes:
        refid: String with name of reference frame (eg. IAU_MOON), string.
        Ra,Rb,Rc: Equatorial radii and polar radius, float, meters.
        f: Flattening parameter, f=(Ra-Rc)/Ra, float.
        mu: Gravitational parameter, GM, float, m^3/s^2
        Prot: Rotational period, float, seconds.
        rhill: Hill-radius with respect to its mater body, float, meters.
        
    State attributes:
        tdb: time of the state.
        Tbod2ecl: transformation matrix from the body r.f. (b.r.f.) to the ecliptic r.f. (e.r.f.)
        Tecl2bod: transformation matrix from the e.r.f. to the b.r.f.
        
    """
    state=np.zeros(6)
    Tbod2ecl=np.zeros((3,3))
    
    def __init__(self,objid):
        self.id=objid
        
        try:
            self.rf=Spice.RF[self.id]
        except Exception as e:
            errorMsg(e,f"Body {self.id} is not available in the package.")
            raise
        
        #Master id and distance to master
        self.master,self.amaster=Spice.Master[self.id]
        
        #Shape of body
        Spice.calcShape(self.id)
        self.Ra=Spice.Ra[self.id]
        self.Rb=Spice.Rb[self.id]
        self.Rc=Spice.Rc[self.id]
        self.f=Spice.f[self.id]
        
        #Physical properties
        self.mu=Spice.Mu[self.id]
        self.Prot=Spice.Prot[self.id]
        Spice.calcHillRadius(self.id)
        self.rhill=Spice.RH[self.id]
        
        #Initialize matrices
        self.Tbod2ecl=None
        self.Tecl2bod=None
                    
    def updateBody(self,tdb):
        """
        Update the state vector with respect to SSB and the transformation matrices.
        
        Parameters:
            tdb: barycentric dynamic time, float, seconds.

        Return: None
        """
        self.tdb=tdb
        state,tlight=spy.spkezr(self.id,tdb,"ECLIPJ2000","NONE","SSB")
        self.stateHelio=Util.transformState(state,[Const.km,Const.km/Const.s])
        self.Tbod2ecl=spy.pxform(self.rf,"ECLIPJ2000",tdb)
        self.Tecl2bod=np.linalg.inv(self.Tbod2ecl)
        
    def calcState(self,tdb):
        """
        Update the state vector with respect to SSB and the transformation matrices.
        
        Parameters:
            tdb: barycentric dynamic time, float, seconds.

        Return: None
        """
        state,tlight=spy.spkezr(self.id,tdb,"ECLIPJ2000","NONE","SSB")
        return Util.transformState(state,[Const.km,Const.km/Const.s])

#################################################################################
#Location class
#################################################################################
class Location(object):
    """
    Define a location on a body.
    
    Initialization atributtes:
        body: Body where the location is, Body class 
        longitud: longitude, float, radians
        latitude: latitude, float, radians
        altitude: elevation over reference ellipsoid, float, m
        
    Secondary attributes:
        posLocal: position of the location with respect to the local r.f.(l.r.f.), numpy array (3), meters.
            NOTE: It is always 0.
        velLocal: velocity of the location with respect to the l.r.f., numpy array (3), meter/seconds.
            NOTE: It is the rotation velocity and it is directed toward east.
        posBody: position of the location with respect to the body r.f. (b.r.f.), numpy array (3), meters.
        velBody: velocity of the location with respect to the b.r.f., numpy array (3), meter/seconds.
        Tlob2bod: transformation matrix from the l.r.f. to the b.r.f.
        Tbod2loc: transformation matrix from the b.r.f. to the l.r.f.
        
    Additional attributes:
        posEcl: velocity of the location with respect to the ecliptic r.f. (e.r.f.), numpy array (3), meters.
        velEcl: velocity of the location with respect to the e.r.f., numpy array (3), meter/seconds.
    
    """
    
    def __init__(self,body,longitude,latitude,altitude):

        #Deep copying is the safest way to get the properties of the body
        self.body=deepcopy(body)
        self.body.updateBody(0)
        self.body.Tecl2bod=None
        self.body.Tbod2ecl=None
        
        #Location on the surface of the body
        self.lon=longitude
        self.lat=latitude
        self.alt=altitude
        
        #Rectangula position
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
    
    def updateLocation(self,tdb):
        """
        Update the state of the location
        
        Parameters:
            tdb: barycentric dynamic time, float, seconds.

        Return: None
        """
        self.body.updateBody(tdb)
        
        #Position of the location in the Ecliptic reference system
        self.posEcl=spy.mxv(self.body.Tbod2ecl,self.posBody)
        
        #Velocity of the location in the Ecliptic reference system
        self.velEcl=spy.mxv(self.body.Tbod2ecl,self.velBody)
    
    def vbod2loc(self,vBod):
        """
        Transform a vector from the body reference frame to its direction on the sky.
        
        Parameters:
            vBod: Vector in the body system, numpy array (3)
            
        Return:
            A: Azimuth (0,2 pi), float, radians
            h: Elevation (-pi,pi), float, radians
            vmag: Magnitude of the vector, float
        """
        vLoc=spy.mxv(self.Tbod2loc,vBod)
        vmag,A,h=spy.reclat(vLoc)
        A=2*np.pi+A if A<0 else A

        #We choose vmag as a negative value using the convention that this are vectors pointing to the observer
        return A,h,-vmag

    def loc2vbod(self,A,h,v):
        """
        Express a vector in the direction A,h with magnitude v in the rotating
        reference frame of the central object of the location.
        
        Parameters:
            A: Azimuth (0,2 pi), float, radians
            h: Elevation (-pi,pi), float, radians
            v: Vector magnitude, (-infty,infty), float, (arbitrary)
                NOTE: If v<0 then the vector points in the opposite direction of (A,h)
                
        Return:
            vBod: Vector in the body-fixed system, np.array, km/s
        """
        vLoc=spy.latrec(v,A,h)
        vBod=spy.mxv(self.Tloc2bod,vLoc)
        return vBod
    
    def ecl2loc(self,eclon,eclat):
        """
        Convert ecliptic coordinates into horizontal coordinates.
        
        Parameters:
            eclon: Ecliptic longitude, float, radians
            eclat: Ecliptic latitude, float, radians
            
        Return:
            A: Azimuth (0,2 pi), float, radians
            h: Elevation (-pi,pi), float, radians
        
        NOTE: It requires to run previously the update method.
        """
        ecx,ecy,ecz=spy.latrec(1,eclon,eclat)
        
        try:
            x,y,z=spy.mxv(self.Tbod2loc,spy.mxv(self.body.Tecl2bod,[ecx,ecy,ecz]))
        except Exception as e:
            errorMsg(e,"You must first update body state.")
            raise
        
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
        
        try:
            ecx,ecy,ecz=spy.mxv(self.body.Tbod2ecl,spy.mxv(self.Tloc2bod,[x,y,z]))
        except Exception as e:
            errorMsg(e,"You must first update body state.")
            raise
                    
        r,eclon,eclat=spy.reclat([ecx,ecy,ecz])
        eclon=2*np.pi+eclon if eclon<0 else eclon
        return eclon,eclat

